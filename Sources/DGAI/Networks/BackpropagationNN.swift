final public class BackpropagationNN {
  static let shared = BackpropagationNN()
  private init() {}
  
  public var learningRate: Float = 0.1
  public var ephos: Int = 5000
  public var weightsInitialRange: Float = .random(in: -4...4)
  
  public var inputLayer: Layer = Layer()
  public var hiddenLayers: [Layer] = []
  public var outputLayer: Layer = Layer()
  
  public func createInputLayer(values: [Float], weights: Int) {
    for value in values {
      let neuron = Neuron()
      neuron.value = value
      neuron.weights = [Float](repeating: weightsInitialRange, count: weights)
      inputLayer.neurons.append(neuron)
    }
  }
  
  public func createHiddenLayer(neurons: Int, weights: Int) {
    let layer = Layer()
    for _ in 0..<neurons {
      let neuron = Neuron()
      neuron.weights = [Float](repeating: weightsInitialRange, count: weights)
      layer.neurons.append(neuron)
    }
    hiddenLayers.append(layer)
  }
  
  public func createOutputLayer(neurons: Int) {
    for _ in 0..<neurons {
      let neuron = Neuron()
      outputLayer.neurons.append(neuron)
    }
  }
  
  public func predict() {}
  
  public func train() {
    
  }
}
