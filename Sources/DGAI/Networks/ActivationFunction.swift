import Foundation

public class ActivationFunction {
  public class func sigmoid(x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
  }
  
  public class func tanh(x: Float) -> Float {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  }
  
  public class func relu(x: Float) -> Float {
    return max(0, x)
  }
}
