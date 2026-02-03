mod neuron;
mod dataset;

use neuron::Neuron;
use dataset::get_dataset;

fn main() {
    let dataset = get_dataset();
    
    for (x, y) in &dataset {
        println!("  ({}, {})", x, y);
    }

    let mut neuron = Neuron::new(0.5, 1.0);

    println!("Entraînement en cours...\n");
    neuron.train(&dataset, 0.01, 10000);

    println!("\nValeurs finales:");
    println!("Weight: {:.4}, Bias: {:.4}", neuron.weight, neuron.bias);
    println!("Loss finale: {:.4}", neuron.loss(&dataset));

    println!("\nTest avec des valeurs INCONNUES (hors dataset):");
    println!("┌────────────┬────────────┬────────────┬────────────┐");
    println!("│     x      │   y_pred   │ y_attendu  │   erreur   │");
    println!("├────────────┼────────────┼────────────┼────────────┤");
    let test_values = vec![0.0, 2.5, 7.5, 16.0, 20.0, 100.0, 382.0, 764.0, 1000.0, 2500.0, 10000.0];
    for x in test_values {
        let y_pred = neuron.forward(x);
        let y_expected = 2.0 * x + 4.0; // Valeur théorique
        let error = (y_pred - y_expected).abs();
        println!("│ {:>10.1} │ {:>10.4} │ {:>10.1} │ {:>10.4} │", 
                 x, y_pred, y_expected, error);
    }
    println!("└────────────┴────────────┴────────────┴────────────┘");
}
