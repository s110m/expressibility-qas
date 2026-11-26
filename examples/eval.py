import argparse
import datetime
import os
import pdb
################ I add the following for erproducibility
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
###################
import torch
import torch.backends.cudnn
import tqdm
import torch.nn.functional as F
import torchquantum as tq

from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from core.tools import StepTimer
from torchquantum.util import (legalize_unitary, build_module_from_op_list,
                                get_v_c_reg_mapping, get_cared_configs,
                                get_success_rate)
from torchquantum.plugin import tq2qiskit, qiskit2tq, tq2qiskit_parameterized
from torchquantum.algorithm.quantumnas.super_utils import get_named_sample_arch
from core.tools.quantize import PACTActivationQuantizer


def log_acc(output_all, target_all, k=1):
    _, indices = output_all.topk(k, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Loss: {loss}")
    return accuracy, loss, size


def save_eval_result(run_dir, filename, payload):
    """Persist an evaluation result to `<run_dir>/eval_results/<filename>.json`.

    eval.py is not wired into torchpack's file-logging sink (only train.py
    calls `set_run_dir`), so without this the accuracy/loss only ever exist in
    console scrollback and re-checking a finished run means re-running the
    whole evaluation. Mirrors the `eval_timing/<filename>.json` layout so a
    given eval config's timing and result sit under the same file name.
    """
    path = os.path.join(run_dir, 'eval_results', f'{filename}.json')
    record = {'timestamp': datetime.datetime.now().isoformat(), **payload}
    try:
        io.save(path, record)
        logger.info(f'Saved eval results to "{path}".')
    except Exception as e:
        logger.warning(f'Could not save eval results: {e}')


def jsonable(value):
    """Best-effort conversion of numpy/torch scalars to plain Python."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, 'item') and getattr(value, 'ndim', None) == 0:
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> None:
    ########################
    import random
    import numpy as np
    import torch

    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)
    #################################

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--print-configs', action='store_true',
                        help='print ALL configs')
    parser.add_argument('--jobs', type=int, default=None,
                        help='max parallel job on qiskit')
    parser.add_argument('--result-suffix', type=str, default=None,
                        help='appended to the eval_results/eval_timing '
                             'filename, so several conditions sharing one eval '
                             'config do not overwrite each other. Needed when '
                             'sweeping qiskit.initial_layout: unlike '
                             'qiskit.ignore_ckpt_layout, a layout passed on the '
                             'command line does not otherwise change the '
                             'output filename.')
    parser.add_argument('--hub', type=str, default=None,
                        help='IBMQ provider')

    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    # capture the seed the checkpoint at --run-dir was actually trained with,
    # since the recursive load below re-pulls in the eval config chain's own
    # ancestor default.yml files (e.g. mnist/default.yml's debug.seed: 42),
    # which would otherwise silently overwrite it. Same failure mode as
    # run_es.py:229-238. This only affects what gets *reported* as train_seed:
    # evaluation itself is deliberately hard-seeded to 0 above, so that two
    # checkpoints are always compared under identical evaluation randomness.
    trained_seed = configs.debug.seed
    trained_set_seed = configs.debug.set_seed
    configs.load(args.config, recursive=True)
    configs.debug.seed = trained_seed
    configs.debug.set_seed = trained_set_seed
    configs.update(opts)

    # for eval, always need load weights
    configs.ckpt.weight_from_scratch = False

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    configs.verbose = args.verbose
    configs.qiskit.hub = args.hub

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.jobs is not None:
        configs.qiskit.max_jobs = args.jobs

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    if args.print_configs:
        print_conf = configs
    else:
        print_conf = get_cared_configs(configs, 'eval')

    logger.info(f'Evaluation started: "{args.run_dir}".' + '\n' +
                f'{print_conf}')

    # R-A control (PAPER_STORY_PURE.md §6): re-evaluate a checkpoint under the
    # transpiler's own initial-layout choice instead of the layout stored in
    # its `solution`, so one trained checkpoint serves both the searched-layout
    # condition and the Qiskit-default-layout baseline. Older run dirs and the
    # non-mnist config trees have no such key, hence the getattr default.
    ignore_ckpt_layout = bool(
        getattr(configs.qiskit, 'ignore_ckpt_layout', False))

    eval_config_dir = args.config.replace('/', '.').replace(
        'examples.', '').replace('.yml', '').replace('configs.', '')
    if ignore_ckpt_layout:
        # separate result/timing file: the two conditions share an eval config,
        # so without this the control would overwrite the searched-layout result
        eval_config_dir = f'{eval_config_dir}.qiskit_layout'
    if args.result_suffix:
        # same need, but for conditions the config tree knows nothing about --
        # chiefly an explicit qiskit.initial_layout override, which changes what
        # is measured without changing the config path this name derives from
        eval_config_dir = f'{eval_config_dir}.{args.result_suffix}'
    configs.eval_config_dir = eval_config_dir
    configs.run_dir = args.run_dir

    # if configs.qiskit.use_qiskit:
    #     IBMQ.load_account()
    #     if configs.run.bsz == 'qiskit_max':
    #         configs.run.bsz = IBMQ.get_provider(hub='ibm-q').get_backend(
    #             configs.qiskit.backend_name).configuration().max_experiments

    with StepTimer('evaluation',
                   out_dir=os.path.join(args.run_dir, 'eval_timing'),
                   filename=f'{eval_config_dir}.json',
                   extra={'config': args.config, 'run_dir': args.run_dir}):
        dataset = builder.make_dataset()
        sampler = torch.utils.data.SequentialSampler(dataset[
                                                         configs.dataset.split])
        dataflow = torch.utils.data.DataLoader(
            dataset[configs.dataset.split],
            sampler=sampler,
            batch_size=configs.run.bsz,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)

        state_dict = io.load(
            os.path.join(args.run_dir, configs.ckpt.name),
            map_location='cpu', weights_only=False)
        model_load = state_dict['model_arch']
        model = builder.make_model()
        for module_load, module in zip(model_load.modules(), model.modules()):
            if isinstance(module, tq.RandomLayer):
                # random layer, need to restore the architecture
                module.rebuild_random_layer_from_op_list(
                    n_ops_in=module_load.n_ops,
                    wires_in=module_load.wires,
                    op_list_in=module_load.op_list,
                )


        model.load_state_dict(state_dict['model'], strict=False)

        # model = state_dict['model_arch']

        # missing, unexpected = model.load_state_dict(
        #     state_dict['model'],
        #     strict=False
        # )

        # if len(missing) > 0 or len(unexpected) > 0:
        #     logger.warning(f"Missing keys: {missing}")
        #     logger.warning(f"Unexpected keys: {unexpected}")

        solution = None
        if 'solution' in state_dict.keys():
            solution = state_dict['solution']
            logger.info(f"Evaluate the solution {solution}")
            logger.info(f"Original score: {state_dict['score']}")
            model.set_sample_arch(solution['arch'])

        if 'v_c_reg_mapping' in state_dict.keys():
            if getattr(model, 'q_layer', None) is not None:
                try:
                    model.measure.set_v_c_reg_mapping(
                        state_dict['v_c_reg_mapping'])
                except AttributeError:
                    logger.warning(f"Cannot set v_c_reg_mapping.")
            elif getattr(model, 'nodes', None) is not None:
                for k, node in enumerate(model.nodes):
                    node.measure.set_v_c_reg_mapping(
                        state_dict['v_c_reg_mapping'][k])

        if state_dict.get('q_layer_op_list', None) is not None and not \
                configs.model.load_op_list:
            logger.warning(f"the model has op_list but is not loaded!!")

        if configs.model.load_op_list:
            assert state_dict['q_layer_op_list'] is not None
            logger.warning(f"Loading the op_list, will replace the q_layer in "
                           f"the original model!")
            if getattr(model, 'q_layer', None) is not None:
                q_layer = build_module_from_op_list(
                    op_list=state_dict['q_layer_op_list'],
                    remove_ops=configs.prune.eval.remove_ops,
                    thres=configs.prune.eval.remove_ops_thres)
                model.q_layer = q_layer
            elif getattr(model, 'nodes', None) is not None:
                for k, node in enumerate(model.nodes):
                    q_layer = build_module_from_op_list(
                        op_list=state_dict['q_layer_op_list'][k],
                        remove_ops=configs.prune.eval.remove_ops,
                        thres=configs.prune.eval.remove_ops_thres)
                    node.q_layer = q_layer

        if state_dict.get('noise_model_tq', None) is not None:
            # the readout error is ALSO applied for eval and test so need load
            # noise_model_tq
            if getattr(model, 'q_layer', None) is not None:
                model.set_noise_model_tq(state_dict['noise_model_tq'])
                if getattr(configs, 'add_noise', False):
                    model.noise_model_tq.mode = 'train'
                    model.noise_model_tq.noise_total_prob = \
                        configs.noise_total_prob
                else:
                    model.noise_model_tq.mode = 'test'
            elif getattr(model, 'nodes', None) is not None:
                for k, node in enumerate(model.nodes):
                    node.set_noise_model_tq(state_dict['noise_model_tq'][k])
                    if getattr(configs, 'add_noise', False):
                        node.noise_model_tq.mode = 'train'
                        node.noise_model_tq.noise_total_prob = \
                            configs.noise_total_prob
                    else:
                        node.noise_model_tq.mode = 'test'

        if configs.model.transpile_before_run:
            # transpile the q_layer
            logger.warning(f"Transpile the q_layer to basis gate set before "
                           f"evaluation, will replace the q_layer!")
            processor = builder.make_qiskit_processor()

            circ = tq2qiskit(model.q_device, model.q_layer)

            """
            add measure because the transpile process may permute the wires,
            so we need to get the final q reg to c reg mapping
            """
            circ.measure(list(range(model.q_device.n_wires)),
                         list(range(model.q_device.n_wires)))

            if solution is not None:
                processor.set_layout(solution['layout'])
                # processor.set_layout([1, 1, 1, 1])
                logger.warning(f"Set layout {solution['layout']} for transpile!")

            logger.info("Transpiling circuit...")
            circ_transpiled = processor.transpile(circs=circ)
            q_layer = qiskit2tq(circ=circ_transpiled)

            model.measure.set_v_c_reg_mapping(
                get_v_c_reg_mapping(circ_transpiled))
            model.q_layer = q_layer

        if configs.legalization.legalize:
            legalize_unitary(model)

        if configs.act_quant.add_in_eval:
            quantizers = []
            assert getattr(model, 'nodes', None) is not None
            if getattr(configs.act_quant, 'act_quant_bit', None) is not None:
                # settings from config file has higher priority
                act_quant_bit = configs.act_quant.act_quant_bit
                act_quant_ratio = configs.act_quant.act_quant_ratio
                act_quant_level = configs.act_quant.act_quant_level
                act_quant_lower_bound = configs.act_quant.act_quant_lower_bound
                act_quant_upper_bound = configs.act_quant.act_quant_upper_bound
                logger.warning(f"Get act_quant setting from config file!")
            elif state_dict.get('act_quant', None) is not None:
                act_quant_bit = state_dict['act_quant']['act_quant_bit']
                act_quant_ratio = state_dict['act_quant']['act_quant_ratio']
                act_quant_level = state_dict['act_quant']['act_quant_level']
                act_quant_lower_bound = state_dict['act_quant'][
                    'act_quant_lower_bound']
                act_quant_upper_bound = state_dict['act_quant'][
                    'act_quant_upper_bound']
                logger.warning(f"Get act_quant setting from ckpt file!")
            elif getattr(configs.trainer, 'act_quant_bit', None) is not None:
                # if the act_quant info is not stored in ckpt, use the info from
                # training config file
                act_quant_bit = configs.trainer.act_quant_bit
                act_quant_ratio = configs.trainer.act_quant_ratio
                act_quant_level = configs.trainer.act_quant_level
                act_quant_lower_bound = configs.trainer.act_quant_lower_bound
                act_quant_upper_bound = configs.trainer.act_quant_upper_bound
                logger.warning(f"Get act_quant setting from previous training "
                               f"config file!")
            else:
                raise NotImplementedError('No act_quant info specified!')

            logger.warning(f"act_quant_bit={act_quant_bit}, "
                           f"act_quant_ratio={act_quant_ratio}, "
                           f"act_quant_level={act_quant_level}, "
                           f"act_quant_lower_bound={act_quant_lower_bound}, "
                           f"act_quant_upper_bound={act_quant_upper_bound}")

            for k, node in enumerate(model.nodes):
                if configs.trainer.act_quant_skip_last_node and k == len(
                        model.nodes) - 1:
                    continue
                quantizer = PACTActivationQuantizer(
                    module=node,
                    precision=act_quant_bit,
                    level=act_quant_level,
                    alpha=1.0,
                    backprop_alpha=False,
                    quant_ratio=act_quant_ratio,
                    device=device,
                    lower_bound=act_quant_lower_bound,
                    upper_bound=act_quant_upper_bound,
                )
                quantizers.append(quantizer)

            for quantizer in quantizers:
                quantizer.register_hook()

        if getattr(configs, 'pre_specified_mean', None) is not None and \
                configs.pre_specified_std \
                is not None:
            for k, node in enumerate(model.nodes):
                node.pre_specified_mean_std = {
                    'mean': configs.pre_specified_mean[k],
                    'std': configs.pre_specified_std[k],
                }

        model.to(device)
        model.eval()

        layout = None
        if configs.qiskit.use_qiskit:
            qiskit_processor = builder.make_qiskit_processor()
            if configs.qiskit.initial_layout is not None:
                layout = configs.qiskit.initial_layout
                logger.warning(f"Use layout {layout} from config file")
            elif ignore_ckpt_layout:
                # R-A control: the checkpoint may carry a searched/manual
                # layout, but we deliberately hand the transpiler a free
                # choice so the run measures Qiskit's own layout heuristic.
                layout = None
                ckpt_layout = state_dict.get('solution', {}).get('layout')
                logger.warning(
                    f"qiskit.ignore_ckpt_layout is set: ignoring checkpoint "
                    f"layout {ckpt_layout} and letting the transpiler choose")
            elif 'solution' in state_dict.keys():
                layout = state_dict['solution']['layout']
                logger.warning(f"Use layout {layout} from checkpoint file")
            else:
                layout = None
                logger.warning(f"No specified layout")
            qiskit_processor.set_layout(layout)
            model.set_qiskit_processor(qiskit_processor)

        if getattr(configs.model.arch, 'sample_arch', None) is not None:
            sample_arch = configs.model.arch.sample_arch
            logger.warning(f"Setting sample arch {sample_arch} from config file!")
            if isinstance(sample_arch, str):
                # this is the name of arch
                sample_arch = get_named_sample_arch(model.arch_space, sample_arch)
                logger.warning(f"Decoded sample arch: {sample_arch}")
            model.set_sample_arch(sample_arch)

        # Provenance shared by every result record below: enough to tell two
        # saved results apart without re-deriving them from the config tree.
        result_meta = {
            'eval_config': args.config,
            'run_dir': args.run_dir,
            'ckpt': configs.ckpt.name,
            'dataset': configs.dataset.name,
            'split': configs.dataset.split,
            # from the run dir's own metainfo (captured before the recursive
            # load), never from the config chain in scope
            'train_seed': trained_seed,
            'use_qiskit': configs.qiskit.use_qiskit,
            'backend_name': getattr(configs.qiskit, 'backend_name', None),
            'n_shots': getattr(configs.qiskit, 'n_shots', None),
            'optimization_level': getattr(
                configs.qiskit, 'optimization_level', None),
            'noise_model_name': getattr(
                configs.qiskit, 'noise_model_name', None),
            'layout': jsonable(layout),
            'ignore_ckpt_layout': ignore_ckpt_layout,
            'solution': jsonable(solution),
            'es_score': jsonable(state_dict.get('score', None)),
        }

        if configs.get_n_params:
            n_params = model.count_sample_params()
            logger.info(f"Number of sampled params: {n_params}")
            save_eval_result(args.run_dir, eval_config_dir, {
                **result_meta,
                'n_sampled_params': n_params,
            })
            exit(0)

        if configs.qiskit.est_success_rate:
            circ_parameterized, params = tq2qiskit_parameterized(
                model.q_device, model.encoder.func_list)
            circ_fixed = tq2qiskit(model.q_device, model.q_layer)
            circ = circ_parameterized + circ_fixed
            transpiled_circ = model.qiskit_processor.transpile(circ)

            success_rate = get_success_rate(
                model.qiskit_processor.properties,
                transpiled_circ)
            logger.info(f"Success rate: {success_rate}")
            logger.info(f"Size: {transpiled_circ.size()}")
            logger.info(f"Depth: {transpiled_circ.depth()}")
            logger.info(f"Width: {transpiled_circ.width()}")
            save_eval_result(args.run_dir, eval_config_dir, {
                **result_meta,
                'success_rate': jsonable(success_rate),
                'circuit_size': transpiled_circ.size(),
                'circuit_depth': transpiled_circ.depth(),
                'circuit_width': transpiled_circ.width(),
            })
            exit(0)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Model Size: {total_params}')

        n_sampled_params = None
        if hasattr(model, 'sample_arch') and not configs.model.load_op_list:
            n_sampled_params = model.count_sample_params()
            logger.info(f"Number of sampled params: {n_sampled_params}")

        with torch.no_grad():
            target_all = None
            output_all = None
            for feed_dict in tqdm.tqdm(dataflow):
                if configs.run.device == 'gpu':
                    inputs = feed_dict[configs.dataset.input_name].cuda(non_blocking=True)
                    targets = feed_dict[configs.dataset.target_name].cuda(
                        non_blocking=True)
                else:
                    inputs = feed_dict[configs.dataset.input_name]
                    targets = feed_dict[configs.dataset.target_name]

                outputs = model(inputs,
                                verbose=configs.verbose,
                                use_qiskit=configs.qiskit.use_qiskit)

                if target_all is None:
                    target_all = targets
                    output_all = outputs
                else:
                    target_all = torch.cat([target_all, targets], dim=0)
                    output_all = torch.cat([output_all, outputs], dim=0)
                # if configs.verbose:
                #     logger.info(f"Measured log_softmax: {outputs}")
                if not configs.dataset.name == 'vqe':
                    log_acc(output_all, target_all)

        logger.info("Final:")
        result = {
            **result_meta,
            'model_size': total_params,
            'n_sampled_params': n_sampled_params,
        }
        if not configs.dataset.name == 'vqe':
            accuracy, loss, n_samples = log_acc(output_all, target_all)
            result.update({
                # accuracy is the fraction, as logged; the percent form is what
                # train.py's scalars.jsonl records, so keep both to make the
                # two directly comparable.
                'accuracy': accuracy,
                'accuracy_percent': accuracy * 100,
                'loss': loss,
                'n_samples': n_samples,
            })
        else:
            eigenvalue = output_all.detach().cpu().numpy()
            logger.info(f"Eigenvalue: {eigenvalue}")
            result['eigenvalue'] = eigenvalue.tolist()

        save_eval_result(args.run_dir, eval_config_dir, result)


if __name__ == '__main__':
    main()
