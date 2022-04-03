public class BackpropagationNN {
  
  public init() {}
  
  public var learningRate: Float = 0.1
  public var ephos: Int = 5000
  public var weightsInitialRange: Float = .random(in: -4...4)
  
  public var inputLayer: Layer = Layer()
  public var hiddenLayers: [Layer] = []
  public var outputLayer: Layer = Layer()
  
  public func createInputLayer(values: [Float], nextLayerNeuronsCount: Int) {
    var index: Int = 0
    for value in values {
      let neuron = Neuron()
      neuron.value = value
      neuron.weights = [Float](repeating: weightsInitialRange, count: nextLayerNeuronsCount)
      print("INPUT LAYER: VALUE - \(value),  WEIGHTS - \(neuron.weights) for \(index + 1) NEURON")
      inputLayer.neurons.append(neuron)
      index += 1
    }
  }
  
  public func createHiddenLayer(neurons: Int, nextLayerNeuronsCount: Int) {
    let layer = Layer()
    for _ in 0..<neurons {
      let neuron = Neuron()
      neuron.weights = [Float](repeating: weightsInitialRange, count: nextLayerNeuronsCount)
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
  
  public func predict() {
    let nextLayerNeurons = hiddenLayers[0].neurons.count
    let h1 = inputLayer.forward(nextLayerNeurons: nextLayerNeurons)
    print(h1)
  }
  
  public func train() {
    
  }
}
