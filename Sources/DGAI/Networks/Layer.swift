public class Layer {
  public var neurons: [Neuron] = []
  
  public func forward(nextLayerNeurons: Int) -> [Float] {
    var result: [Float] = []
    for i in 0..<nextLayerNeurons {
      var value: Float = 0
      for n in neurons {
        value = n.value * n.weights[i]
      }
      result.append(value)
    }
    return result
  }
}
