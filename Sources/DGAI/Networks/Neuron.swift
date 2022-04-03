public class Neuron {
  public var value: Float = 0
  public var weights: [Float] = []
  
  public func forward() -> [Float] {
    return weights.map { $0 * value }
  }
}
