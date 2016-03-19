#![feature(iter_arith)]
extern crate collenchyma as co;
extern crate leaf;
extern crate rand;

use std::cmp::min;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use rand::distributions::{IndependentSample, Range};

use co::prelude::{Backend, IBackend, Cuda, Native, SharedTensor, ITensorDesc};
use leaf::layer::{LayerConfig, LayerType};
use leaf::layers::{SequentialConfig, LinearConfig, NegativeLogLikelihoodConfig};
use leaf::solver::{Solver, SolverConfig};

fn print_tensor<T: std::fmt::Debug>(x: &mut SharedTensor<T>) {
    let native = Backend::<Native>::default().unwrap();
    // x.add_device(native.device()).unwrap();
    x.sync(native.device()).unwrap();
    let mem = x.get(native.device()).unwrap().as_native().unwrap().as_slice::<T>();
    let dims = x.desc().dims();
    println!("Dims={:?}, mem={:?}", dims, mem);
}

fn fill_random(xs: &mut [f32]) {
    let range = Range::new(-1.0, 1.0);
    let mut rng = rand::thread_rng();
    for x in xs {
        *x = range.ind_sample(&mut rng);
    }
}

fn classify(xs: &[f32], classes_n: usize) -> usize {
    // let r2 = xs.iter().map(|x| x * x).sum::<f32>() / xs.len() as f32;
    // let class = r2.sqrt() * (classes_n as f32 + 1.0);
    // min(classes_n - 1, class as usize)
    let x = xs[0] * 0.5 + 0.5;
    let class = x * (classes_n as f32);
    println!("input: {} {}", x, class as usize);
    min(classes_n - 1, class as usize)
}


fn prepare_minibatch(input: &mut SharedTensor<f32>, labels: &mut SharedTensor<f32>,
                     classes_n: usize) -> Vec<usize> {
    let mut targets = Vec::new();

    let batch_size = input.desc().dims()[0];
    let inp_size = input.desc().dims()[1];

    let native = Backend::<Native>::default().unwrap();
    input.sync(native.device()).unwrap();
    let input_mem = input.get_mut(native.device()).unwrap()
        .as_mut_native().unwrap().as_mut_slice::<f32>();
    let labels_mem = labels.get_mut(native.device()).unwrap()
        .as_mut_native().unwrap().as_mut_slice::<f32>();

    for batch_i in 0..batch_size {
        let mut xs = &mut input_mem[inp_size * batch_i ..
                                    inp_size * (batch_i + 1)];
        fill_random(xs);
        let label = classify(xs, classes_n);
        labels_mem[batch_i] = label as f32;
        targets.push(label);
    }
    targets
}

fn main() {
    let batch_size: usize = 1;
    let input_size: usize = 1;
    let output_size: usize = 5;
    let learning_rate = 0.0000f32;
    let momentum = 0f32;

    let mut net_cfg = SequentialConfig::default();
    net_cfg.force_backward = true;

    net_cfg.add_input("data", &[batch_size, input_size]);
    net_cfg.add_layer(LayerConfig::new(
        "linear1", LayerType::Linear(LinearConfig { output_size: 10 })));
    net_cfg.add_layer(LayerConfig::new(
        "sigmoid", LayerType::Sigmoid));
    net_cfg.add_layer(LayerConfig::new(
        "linear2", LayerType::Linear(LinearConfig { output_size: 10 })));
    net_cfg.add_layer(LayerConfig::new(
        "relu", LayerType::ReLU));
    net_cfg.add_layer(LayerConfig::new(
        "linear3", LayerType::Linear(LinearConfig { output_size: output_size })));

    let mut classifier_cfg = SequentialConfig::default();
    classifier_cfg.add_input("network_out", &vec![batch_size, output_size]);
    classifier_cfg.add_input("label", &vec![batch_size, 1]);
    classifier_cfg.add_layer(LayerConfig::new(
        "nll", LayerType::NegativeLogLikelihood(
            NegativeLogLikelihoodConfig { num_classes: output_size })));

    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum: momentum,
        .. SolverConfig::default()
    };
    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);

    let mut confusion = ::leaf::solver::ConfusionMatrix::new(output_size);
    confusion.set_capacity(Some(1000));

    let backend = Rc::new(Backend::<Cuda>::default().unwrap());
    let native_backend = Rc::new(Backend::<Native>::default().unwrap());

    let mut solver = Solver::from_config(backend.clone(), backend.clone(),
                                         &solver_cfg);

    let inp_lock = Arc::new(RwLock::new(SharedTensor::<f32>::new(
        native_backend.device(), &(batch_size, input_size)).unwrap()));
    let label_lock = Arc::new(RwLock::new(SharedTensor::<f32>::new(
        native_backend.device(), &(batch_size, 1)).unwrap()));
    inp_lock.write().unwrap().add_device(backend.device()).unwrap();

    for i in 0..100000 {
        println!("--------------");
        let targets = prepare_minibatch(
            &mut inp_lock.write().unwrap(),
            &mut label_lock.write().unwrap(),
            output_size);

        // print_tensor(&mut inp_lock.write().unwrap());
        // print_tensor(&mut label_lock.write().unwrap());

        // println!("inp dim before: {:?}", inp_lock.read().unwrap().desc().dims());
        // println!("{:#?}", solver.network());
        for weights in &solver.network().weights_data {
            print_tensor(&mut weights.write().unwrap());
        }

        // println!("net: {:?}", solver.network());

        let infered_out = solver.train_minibatch(inp_lock.clone(),
                                                 label_lock.clone());
        // println!("inp dim after: {:?}", inp_lock.read().unwrap().desc().dims());

        let mut infered = infered_out.write().unwrap();
        let predictions = confusion.get_predictions(&mut infered);

        confusion.add_samples(&predictions, &targets);
        if i % 1 == 0 {
            println!("Last sample: {} | Accuracy {}",
                    confusion.samples().iter().last().unwrap(),
                    confusion.accuracy());
            print_tensor(&mut infered);
        }
    }
        // print_tensor(&mut inp_lock.write().unwrap());
}
