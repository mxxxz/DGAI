public class BackpropagationNN {
  
  public init() {}
  
  public var learningRate: Float = 0.1
  public var ephos: Int = 5000
  public var weightRange: Float = .random(in: -1...1)
  
  public var inputLayer: Layer = Layer()
  public var hiddenLayer: Layer = Layer()
  public var outputLayer: Layer = Layer()
  
  public func createInputLayer(neurons: Int, nextLayerNeuronsCount: Int) {
    for _ in 0..<neurons {
      let neuron = Neuron()
      for _ in 0..<nextLayerNeuronsCount {
        neuron.weights.append(weightRange)
      }
      inputLayer.neurons.append(neuron)
    }
  }
  
  public func createHiddenLayer(neurons: Int, nextLayerNeuronsCount: Int) {
    for _ in 0..<neurons {
      let neuron = Neuron()
      for _ in 0..<nextLayerNeuronsCount {
        neuron.weights.append(weightRange)
      }
      hiddenLayer.neurons.append(neuron)
    }
  }
  
  public func createOutputLayer() {
    outputLayer.neurons.append(Neuron())
  }
  
  public func predict(values: [Float]) -> [Float] {
    return forward(inputValues: values)
  }
  
  public func train(values: [Float], expected: Float) {
    let forward = forward(inputValues: values)
    guard let actual = forward.first else { return }
    let error: Float = actual - expected
    let sigmoidDx = actual * (1 - actual)
    let weightsDelta = error * sigmoidDx
    hiddenLayer.reverse(weightDelta: weightsDelta, learningRate: learningRate)
    for i in 0..<inputLayer.neurons.count {
      for w in 0..<inputLayer.neurons[i].weights.count {
        inputLayer.neurons[i].weights[w] = inputLayer.neurons[i].weights[w] - inputLayer.neurons[i].value * hiddenLayer.neurons[w].weightsDelta * learningRate
      }
    }
  }
  
  private func forward(inputValues: [Float]) -> [Float] {
    inputLayer.set(values: inputValues)
    let forwardResults = inputLayer.forward(nextLayerNeurons: hiddenLayer.neurons.count)
    let sigmoidValues = forwardResults.map { ActivationFunction.sigmoid(x: $0) }
    hiddenLayer.set(values: sigmoidValues)
    let finalResults = hiddenLayer.forward(nextLayerNeurons: outputLayer.neurons.count)
    let finalSigmoidResults = finalResults.map { ActivationFunction.sigmoid(x: $0) }
    outputLayer.set(values: finalSigmoidResults)
    return finalSigmoidResults
  }
}
