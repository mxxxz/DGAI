public class Layer {
  public var neurons: [Neuron] = []
  
  public func set(values: [Float]) {
    for i in 0..<values.count {
      neurons[i].value = values[i]
    }
  }
  
  public func forward(nextLayerNeurons: Int) -> [Float] {
    var result: [Float] = []
    for i in 0..<nextLayerNeurons {
      var value: Float = 0
      for n in neurons {
        value += n.value * n.weights[i]
      }
      result.append(value)
    }
    return result
  }
  
  public func backward(weightDelta: Float, learningRate: Float) {
    for n in neurons {
      for i in 0..<n.weights.count {
        n.weights[i] = n.weights[i] - n.value * weightDelta * learningRate
        n.error = n.weights[i] * weightDelta
        n.weightsDelta = n.error * (n.value * (1 - n.value))
      }
    }
  }
}
