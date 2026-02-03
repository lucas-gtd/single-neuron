// Implémentation d'un neurone simple
// fonction de perte: erreur quadratique moyenne
// optimisation: descente de gradient
// modèle linéaire: y = weight * x + bias
// objectif: apprendre à approximer la fonction "y = 2x + 4"

pub struct Neuron {
    pub weight: f32,
    pub bias: f32,
}

impl Neuron {
    pub fn new(weight: f32, bias: f32) -> Self {
        Neuron { weight, bias }
    }

    pub fn forward(&self, x: f32) -> f32 {
        self.weight * x + self.bias
    }

    pub fn loss(&self, dataset: &[(f32, f32)]) -> f32 {
        let mut total_error = 0.0;
        for (x, y_true) in dataset {
            let y_pred = self.forward(*x);
            let error = y_pred - y_true;
            total_error += error * error;
        }
        total_error / dataset.len() as f32
    }

    pub fn train(&mut self, dataset: &[(f32, f32)], learning_rate: f32, epochs: usize) {
        for epoch in 0..epochs {
            let mut grad_weight = 0.0;
            let mut grad_bias = 0.0;

            for (x, y_true) in dataset {
                let y_pred = self.forward(*x);
                let error = y_pred - y_true;
                grad_weight += 2.0 * error * x;
                grad_bias += 2.0 * error;
            }

            grad_weight /= dataset.len() as f32;
            grad_bias /= dataset.len() as f32;

            self.weight -= learning_rate * grad_weight;
            self.bias -= learning_rate * grad_bias;

            if (epoch + 1) % 100 == 0 {
                let current_loss = self.loss(dataset);
                println!("Epoch {}: Loss = {:.4}, Weight = {:.4}, Bias = {:.4}", 
                         epoch + 1, current_loss, self.weight, self.bias);
            }
        }
    }
}