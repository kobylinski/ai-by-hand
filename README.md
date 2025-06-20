## Week 1: Linear Regression I

**Focus:** Data generation and prediction with a linear function

### Session 1.1: Generate Synthetic Linear Data

* ~~**Goal:** Produce noisy (x, y) data points following a linear pattern~~
* ~~**Environment:** Rust (rand crate)~~
* ~~**Input:** None~~
* ~~**Description:** Write a function that generates 50 x values in a given range (e.g., 0 to 10), and calculates y = 2x + 3 + noise, where noise is a small random value. Print the result as two arrays and optionally export to a CSV or JSON file.~~

### Session 1.2: Linear Prediction Function

* ~~**Goal:** Implement a forward function y = wx + b and compute MSE~~
* ~~**Environment:** Rust~~
* ~~**Input:** Data from Session 1.1~~
* ~~**Description:** Build a function that takes x, w, b and computes y_pred, then calculates Mean Squared Error between y_pred and actual y. Print the loss. Try with various initial values.~~

### Session 1.3: Train with Gradient Descent

* ~~**Goal:** Adjust w and b using gradient descent to reduce MSE~~
* ~~**Environment:** Rust~~
* ~~**Input:** Code and data from previous sessions~~
* ~~**Description:** Implement gradient formulas for linear regression and update w, b over several epochs. Log loss every few steps. Observe convergence.~~

## Week 2: Linear Regression II

* **Focus:** Tuning training process and language translation

### Session 2.1: Learning Rate & Iteration Experiments

* **Goal:** Visualize and compare effects of different learning rates
* **Environment:** Rust + PHP (for contrast)
* **Input:** Code from Session 1.3
* **Description:** Run training with high, low, and ideal learning rates. Note convergence or divergence. Rewrite gradient update logic in PHP and observe differences in numeric handling.

### Session 2.2: Model Reuse Across Languages

* **Goal:** Use Rust-trained model in a PHP or JS prediction app
* **Environment:** Rust → PHP or JS
* **Input:** Trained values for w and b
* **Description:** Export the final model weights. In a new language (e.g., PHP), reimplement the prediction formula and validate that results match those from Rust. Optionally prepare a minimal input UI in HTML + JS.

## Week 3–4: Single Neuron Classification (Logistic Regression)

* **Focus:** Binary classification with one neuron

### Session 3.1: Create a 2D Classification Dataset

* **Goal:** Generate and visualize a synthetic 2D dataset for binary classification
* **Environment:** JavaScript (Node.js + Canvas for optional visualization)
* **Input:** None
* **Description:** Generate two clusters of (x, y) points using different centers (e.g. (1,1) and (5,5)). Assign binary labels. Store data as a JSON array. Optionally display using a basic canvas or HTML visualizer.

### Session 3.2: Implement Logistic Neuron

* **Goal:** Calculate prediction from sigmoid-activated neuron with 2 inputs
* **Environment:** Rust
* **Input:** Dataset from Session 3.1 (load JSON file)
* **Description:** Implement forward pass: z = w1*x + w2*y + b, then sigmoid(z). Use this to generate predictions for each point. Print predicted probabilities and compare to labels.

### Session 3.3: Train Using Cross-Entropy Loss

* **Goal:** Learn neuron weights using binary cross-entropy gradient descent
* **Environment:** Rust
* **Input:** Code and data from Session 3.2
* **Description:** Implement loss function and gradient formulas. Train with mini-batch or full-batch updates over multiple epochs. Monitor and log accuracy or loss.

### Session 3.4: Visualize Decision Boundary

* **Goal:** Show the linear decision line of a trained logistic neuron
* **Environment:** JavaScript (Canvas or HTML)
* **Input:** Trained weights from Rust
* **Description:** Reimplement the prediction formula in JS. Overlay decision boundary (w1*x + w2*y + b = 0) on top of the scatter plot. Highlight classification regions.

### Session 3.5: Reuse Model in JavaScript

* **Goal:** Use model trained in Rust for real-time classification in browser
* **Environment:** JavaScript (HTML + browser runtime)
* **Input:** Exported weights and bias from Rust
* **Description:** Create a small UI where user clicks a point, and the app predicts which class it belongs to. Reuse JS formula from Session 3.4 and add interactive feedback.

### Session 4.1: Re-implement Sigmoid Neuron in PHP

* **Goal:** Translate prediction code to PHP and validate parity with Rust
* **Environment:** PHP (CLI script)
* **Input:** JSON dataset and weights
* **Description:** Implement the same prediction and sigmoid logic in PHP. Feed in sample points and compare predictions to Rust/JS versions. Use this to explore numerical handling differences.

### Session 4.2: Train with Slightly Noisy Data

* **Goal:** Observe logistic regression behavior on less separable data
* **Environment:** Rust
* **Input:** New dataset with overlapping class points
* **Description:** Add Gaussian noise to class centers to create overlap. Retrain and observe how accuracy and loss stabilize. This introduces the concept of "good enough" decision boundaries when data is noisy.

### Session 4.3: Wrong Activation Function Test

* **Goal:** Replace sigmoid with ReLU or identity and observe failure
* **Environment:** Rust
* **Input:** Same training code
* **Description:** Swap sigmoid with a non-saturating function like identity (i.e., f(z)=z) or ReLU. Retrain on binary classification and observe how loss behaves and predictions fail. This teaches the role of activation choice.

### Session 4.4: Logistic Regression with One-Hot Labels

* **Goal:** Encode binary classes as [1,0] and [0,1] and prepare for multi-class
* **Environment:** PHP or Python
* **Input:** Dataset from earlier sessions
* **Description:** Instead of scalar labels, format them as one-hot vectors. Although overkill for binary, this sets up future models to handle multi-class tasks. Write PHP or Python code to convert labels and prepare vectorized targets.

## Week 5–6: Multi-layer Networks and XOR

* **Focus:** Learn non-linear functions with hidden layers

### Session 5.1: Build XOR Dataset

* **Goal:** Generate and print XOR truth table with expected outputs
* **Environment:** PHP
* **Input:** None
* **Description:** Manually define 4 inputs: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0. Store as JSON or array structure. Output to console and/or file.

### Session 5.2: 2-Layer Network Forward Pass

* **Goal:** Calculate output of a 2-layer neural network with sigmoid activations
* **Environment:** PHP
* **Input:** XOR dataset
* **Description:** Define weights for a 2-2-1 network. Implement sigmoid(x), compute hidden layer activations, and then the output. Use fixed weights to verify manual calculations match expected behavior.

### Session 5.3: Derive Backpropagation by Hand

* **Goal:** Calculate gradients of weights for the XOR network manually
* **Environment:** Python (Jupyter Notebook recommended)
* **Input:** Weights, biases, and XOR dataset
* **Description:** Pick one input (e.g. (1,0)) and work through forward pass, compute output error, then backpropagate using chain rule. Print every gradient step to understand how hidden and output weights are affected.

### Session 5.4: Train XOR Network with Backpropagation

* **Goal:** Implement full backpropagation training loop to solve XOR
* **Environment:** Rust
* **Input:** XOR dataset (converted from PHP to JSON or hardcoded)
* **Description:** Implement a training loop for a 2-2-1 network using sigmoid activations. Use mean squared error or binary cross-entropy loss. Initialize random weights. Train until predictions are near 0 or 1 for all inputs.

### Session 5.5: Visual XOR Solver in JavaScript

* **Goal:** Port XOR model to JavaScript and show input/output
* **Environment:** JavaScript (HTML + JS)
* **Input:** Final weights and biases from Rust
* **Description:** Create an HTML UI to let user click 0/1 inputs and display predicted output using the JavaScript port of the trained model. Print the prediction to the screen.

### Session 6.1: Visualize Hidden Neuron Behavior

* **Goal:** Show intermediate outputs of hidden neurons in XOR network
* **Environment:** JavaScript (in browser)
* **Input:** Same model from Session 5.5
* **Description:** Extend the HTML interface to show not only the final output but also the hidden layer outputs. Let the user tweak weights and observe how hidden activations change.

### Session 6.2: XOR with Tanh Activation

* **Goal:** Re-implement XOR network using tanh instead of sigmoid
* **Environment:** Python or PHP
* **Input:** Previous XOR dataset
* **Description:** Replace sigmoid activation with tanh and update gradient formulas. Compare training stability and output behavior.

### Session 6.3: Overfitting Small Network (Experiment)

* **Goal:** Train a larger-than-needed XOR network and inspect behavior
* **Environment:** Rust
* **Input:** XOR dataset
* **Description:** Use a 2-4-4-1 architecture. Add more neurons and layers than needed. Log loss per epoch. Observe slower convergence, unstable gradients, or signs of overfitting.

### Session 6.4: Model Portability – Rust to PHP Reuse

* **Goal:** Use Rust-trained weights in PHP to predict XOR outputs
* **Environment:** Rust → PHP
* **Input:** Weights + biases exported as JSON
* **Description:** Create PHP prediction function. Confirm that XOR predictions using Rust-trained values match across both languages. Reflect on language differences in math precision.

### Session 6.5: Add Noise to XOR Inputs

* **Goal:** Observe how XOR network handles noisy or off-grid inputs
* **Environment:** JavaScript
* **Input:** Existing XOR prediction logic in JS
* **Description:** Slightly alter inputs (e.g., use (0.1, 0.9) instead of (0,1)). See if the network generalizes correctly. Visualize misclassifications if present.

## Week 7–8: Training Experiments

* **Focus:** Architecture, activation functions, under/overfitting

### Session 7.1: Add Hidden Layers to XOR Network

* **Goal:** Create a 3-layer network and test its ability to solve XOR
* **Environment:** Rust
* **Input:** XOR dataset and existing 2-layer model
* **Description:** Extend your XOR solver with an additional hidden layer (e.g., 2–3–2–1). Initialize weights, update forward and backpropagation logic, and test if it still converges correctly. Log and compare final performance.

### Session 7.2: Visualize Training Curve

* **Goal:** Plot training loss over epochs for a classification task
* **Environment:** JavaScript (Chart.js or Canvas)
* **Input:** Training log (loss per epoch) from Rust model

* **Description:** Export loss values to a JSON file and visualize them as a line chart in a browser. See how quickly or slowly convergence happens with different learning rates or model sizes.

### Session 7.3: Underfitting vs Overfitting Exploration

* **Goal:** Compare model performance on training and test data
* **Environment:** PHP
* **Input:** Classification dataset from previous sessions
* **Description:** Split dataset into 70/30 train/test. Train the model on the training portion. Monitor accuracy on both sets over time. Try a too-simple model (underfit) and a large model (overfit) and note the test accuracy trends.

### Session 7.4: Break It – No Activation Function

* **Goal:** Remove hidden layer activations and observe network failure
* **Environment:** Rust
* **Input:** XOR network
* **Description:** Replace sigmoid/ReLU with identity activation (f(x)=x). Retrain the XOR model and see how the network fails to converge. Log weights and losses. Learn why non-linearity is necessary.

### Session 7.5: Try ReLU vs Sigmoid Comparison

* **Goal:** Compare training dynamics with different hidden activations
* **Environment:** Python (Jupyter or script)
* **Input:** XOR or classification dataset
* **Description:** Train one model using sigmoid and another using ReLU. Track number of epochs until convergence and loss/accuracy. Analyze if ReLU converges faster or is more stable in deeper architectures.

### Session 8.1: Varying Learning Rate Experiment

* **Goal:** Measure training quality with different learning rates
* **Environment:** Rust
* **Input:** XOR dataset
* **Description:** Train the same model with α = 0.01, 0.1, 1.0. Log losses and final performance. Identify divergence or slow learning.

### Session 8.2: Weight Initialization Test

* **Goal:** Evaluate impact of initial weights on training
* **Environment:** Rust or Python
* **Input:** XOR dataset
* **Description:** Try initializing all weights to 0, small random, or large values. Observe which ones cause failure or slow learning. Reflect on symmetry breaking and gradient flow.

### Session 8.3: Activation Saturation Demo

* **Goal:** Show how sigmoid saturation slows down learning
* **Environment:** Python (with graph plotting)
* **Input:** Sigmoid function and gradients
* **Description:** Plot sigmoid curve and its derivative. Show that extreme inputs yield near-zero gradients. Train on input that causes this and see slow/no learning. Discuss vanishing gradients.

### Session 8.4: Model Debug Mode – Log Internals

* **Goal:** Add internal state logging for training introspection
* **Environment:** PHP or Rust
* **Input:** Any small model (XOR or classifier)
* **Description:** Log hidden activations, deltas, weight updates per epoch. Use this to debug learning problems and understand backprop in motion.

### Session 8.5: Early Stopping Based on Validation Loss

* **Goal:** Stop training when validation loss starts rising
* **Environment:** Python
* **Input:** Small dataset split into train/validation
* **Description:** Train model, monitor training and validation loss. Stop early if validation loss increases over time (a sign of overfitting). Implement simple logic to track and stop training early.

## Week 9–10: Image Data (Basic Vision)

* **Focus:** Train on small 2D binary image patterns

### Session 9.1: Create 5x5 Binary Image Dataset

* **Goal:** Generate grid representations of symbols like 'X' and 'O'
* **Environment:** Python
* **Input:** None
* **Description:** Define binary matrices (5x5) to represent distinct shapes such as 'X', 'O', or '+' using 0s and 1s. Create multiple variants by rotating or slightly altering the shape. Save them as labeled arrays or JSON for later reuse.

### Session 9.2: Flatten Images for Input

* **Goal:** Convert 2D image matrices into flat input vectors
* **Environment:** Rust
* **Input:** JSON dataset from Session 9.1
* **Description:** Load the 5x5 image matrix and flatten it into a 25-element input vector. Each vector will represent a training example. Create an interface to feed multiple vectors and their associated labels.

### Session 9.3: Single-Layer Image Classifier

* **Goal:** Train a perceptron to distinguish between two image classes
* **Environment:** Rust
* **Input:** Flattened image data from Session 9.2
* **Description:** Build a simple logistic classifier with 25 input weights and a single output. Train using gradient descent to distinguish between 'X' and 'O'. Evaluate on a few test inputs.

### Session 9.4: Multi-Layer Image Classifier

* **Goal:** Extend image model to a 25-5-1 neural network
* **Environment:** Rust
* **Input:** Same dataset
* **Description:** Add a hidden layer with 5 neurons and use sigmoid activation. Implement forward pass and backpropagation. Compare training speed and accuracy to the single-layer model.

### Session 9.5: Port Image Classifier to PHP

* **Goal:** Use Rust-trained weights in PHP to classify new images
* **Environment:** Rust → PHP
* **Input:** Trained model weights and image input data
* **Description:** Export weights and implement the same forward pass logic in PHP. Use a simple CLI interface to test image predictions and verify results match Rust output.

### Session 10.1: Draw Image in Browser and Predict

* **Goal:** Create a UI to input 5x5 grids and classify the pattern
* **Environment:** JavaScript (HTML + Canvas or Grid UI)
* **Input:** None, or user input from the interface
* **Description:** Build a browser interface that lets the user fill out a 5x5 grid (click to toggle black/white). Convert this grid to a vector and run it through a JS version of the classifier model to predict if it’s an 'X' or an 'O'.

### Session 10.2: Add Noise and Test Robustness

* **Goal:** Evaluate model prediction when image pixels are slightly changed
* **Environment:** JavaScript
* **Input:** Classifier and UI from Session 10.1
* **Description:** Randomly flip 1–2 pixels in the user's input and show how it affects the classification result. This tests robustness and shows model sensitivity.

### Session 10.3: Visualize Hidden Activations (Optional)

* **Goal:** Show hidden neuron activations in real-time as user draws
* **Environment:** JavaScript (Canvas + DOM)
* **Input:** Same model from earlier session
* **Description:** Extend the UI to display the activation values of each hidden neuron (from Session 9.4) when the user inputs a new pattern. Helps make the black box more transparent.

### Session 10.4: Add Third Class (e.g. '+')

* **Goal:** Extend dataset and classifier to handle a new symbol
* **Environment:** Python (data), Rust (model)
* **Input:** Add training examples for a third pattern
* **Description:** Update dataset with '+' images. Modify model output layer to support multi-class classification using one-hot encoding and softmax (or 3 sigmoid neurons). Train and test the extended classifier.

### Session 10.5: Misclassification Heatmap

* **Goal:** Visualize which input pixels most affect prediction
* **Environment:** Python (optional), JavaScript (preferred)
* **Input:** Trained model and test input
* **Description:** For a given image, highlight which pixels cause the largest change in output when toggled. Useful for introspection and debugging which features matter most.

## Week 11–12: Text Classification

* **Focus:** Preprocess and learn from sentences

### Session 11.1: Create a Fake Spam Detection Dataset

* **Goal:** Prepare a small set of labeled messages for binary classification
* **Environment:** PHP
* **Input:** None
* **Description:** Manually write ~10 short messages (half spam, half non-spam). Label and store them in a JSON or CSV format. Add basic preprocessing like lowercasing and punctuation removal.

### Session 11.2: Bag-of-Words Feature Encoding

* **Goal:** Convert text to fixed-size numeric feature vectors
* **Environment:** Python
* **Input:** Dataset from Session 11.1
* **Description:** Build a vocabulary of all words in the dataset. Encode each message as a vector where each position indicates word presence (1/0). Output these vectors and corresponding labels to JSON.

### Session 11.3: Reimplement Encoder in PHP

* **Goal:** Translate bag-of-words logic to PHP for comparison
* **Environment:** PHP
* **Input:** Dataset and vocabulary from Session 11.2
* **Description:** Implement the same encoding logic in PHP and validate output vectors against Python results. This highlights implementation differences and strengthens multi-language understanding.

### Session 11.4: Logistic Classifier on Word Vectors

* **Goal:** Train a simple classifier to predict spam
* **Environment:** Rust
* **Input:** Encoded vectors + labels from JSON
* **Description:** Build a model with as many inputs as vocabulary size, and one sigmoid output. Train using binary cross-entropy. Evaluate accuracy on all samples and log learned weights.

### Session 11.5: Visualize Word Influence

* **Goal:** Show which words are most influential for spam prediction
* **Environment:** JavaScript (HTML/DOM)
* **Input:** Trained model weights from Rust
* **Description:** Load weights and vocabulary into JS. Create a table or word cloud where weight magnitude and sign determine word color or size (e.g., red for high-spam words, blue for ham). Helps explain what the model learned.

### Session 12.1: Add Stopword Filtering

* **Goal:** Improve feature quality by excluding common words
* **Environment:** Python
* **Input:** Original message dataset
* **Description:** Implement basic stopword removal (e.g., "the", "is", "and"). Re-encode messages without stopwords. Compare final accuracy of model trained on original vs filtered data.

### Session 12.2: Introduce Typos and Noise in Text

* **Goal:** Observe model robustness to imperfect inputs
* **Environment:** PHP or Python
* **Input:** Original spam dataset
* **Description:** Add spelling errors or extra whitespace into message variants. Re-encode and test classifier accuracy. Discuss how real-world noise degrades performance.

### Session 12.3: Add Neutral Third Class (e.g., Reminder)

* **Goal:** Convert binary to multi-class classification
* **Environment:** Python
* **Input:** Add new message samples with third label
* **Description:** Add a “neutral” message class (e.g., reminders). Change model output to softmax with 3 outputs. Retrain on new data and evaluate accuracy per class.

### Session 12.4: Confusion Matrix Analysis

* **Goal:** Visualize where the model makes classification mistakes
* **Environment:** JavaScript
* **Input:** Predictions vs. true labels
* **Description:** Create a simple confusion matrix table in the browser. Count true positive/negative and false positive/negative predictions. Useful for understanding imbalanced classification issues.

### Session 12.5: Cross-language Prediction Port

* **Goal:** Use model trained in Rust to classify messages in PHP
* **Environment:** Rust → PHP
* **Input:** Weights and vocab from Rust training
* **Description:** Implement PHP version of logistic prediction using trained weights. Accept message input from CLI, encode it, and output classification. Demonstrates portability and reinforces logic consistency.

## Week 13–14: Streaming and Sequential Data

* **Focus:** Online learning + adapting over time

### Session 13.1: Simulate a Simple Data Stream

* **Goal:** Generate a numeric stream with noise and trends
* **Environment:** Rust
* **Input:** None
* **Description:** Create a sequence of numbers representing a time series (e.g., y[t] = 0.5 * y[t-1] + noise). Log data point by point. Optionally simulate sensor-style stream where new data arrives over time.

### Session 13.2: Implement Online Learning (Single Weight)

* **Goal:** Train a simple model one step at a time from a stream
* **Environment:** Rust
* **Input:** Stream generator from Session 13.1
* **Description:** Use a single-feature linear predictor y_pred = w * x_prev + b. After each point, update weights with gradient from just that point’s error. Log prediction accuracy over time.

### Session 13.3: Introduce Regime Shift

* **Goal:** Test model adaptation to a sudden pattern change
* **Environment:** Python
* **Input: Modify stream: halfway through change pattern (e.g., from upward trend to downward)
* **Description:** Observe how quickly the model adapts to the new data regime. Plot prediction vs. actual over time. Show how learning rate and memory affect tracking.

### Session 13.4: Stream Model Port to PHP

* **Goal:** Re-implement online predictor in PHP
* **Environment:** PHP
* **Input:** Live stream or pre-recorded CSV
* **Description:** Create CLI tool in PHP to read stream and update a simple model after each line. Echo predictions and current model state to console. Demonstrates language parity.

### Session 13.5: Persist and Reload Online Model

* **Goal:** Save and resume online-trained model
* **Environment:** PHP
* **Input:** JSON file with current model weights and bias
* **Description:** Extend the PHP script to write the model state to disk periodically and allow it to resume from that state later. Simulates persistence across sessions.

### Session 14.1: Build Live JS Dashboard for Predictions

* **Goal:** Visualize predictions and errors in real time
* **Environment:** JavaScript (HTML + WebSockets or polling)
* **Input:** PHP stream output (via file or server push)
* **Description:** Use JavaScript to read incoming prediction results and plot them on a line graph. Show actual vs predicted values live.

### Session 14.2: Drift Detection Visual

* **Goal:** Highlight model error spikes during data shifts
* **Environment:** JavaScript
* **Input:** Prediction stream
* **Description:** Extend the JS dashboard to compute and display moving average of error. Flag spikes in prediction error visually to signal concept drift.

### Session 14.3: Add Rolling Window Smoothing

* **Goal:** Stabilize predictions using moving average of past weights
* **Environment:** Rust
* **Input:** Stream model
* **Description:** Modify online learner to keep a window of past weights or predictions and average them to stabilize noisy training. Compare raw vs smoothed output.

### Session 14.4: Multi-Sensor Online Learner

* **Goal:** Expand model to predict based on 2+ input streams
* **Environment:** Rust
* **Input:** Simulate multiple sensor values (e.g., temperature + humidity)
* **Description:** Extend the model to have multiple weights. Predict one target value from multiple inputs. Train online and evaluate feature influence.

### Session 14.5: Auto-Learning Rate Adjuster

* **Goal:** Implement learning rate decay or adaptation
* **Environment:** Rust or Python
* **Input:** Any streaming setup
* **Description:** Add logic to modify learning rate dynamically (e.g., reduce over time or increase when error spikes). Evaluate convergence and stability.