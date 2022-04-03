import CoreGraphics

public func MSE(predicts: [Float], expences: [Float]) -> Float {
  var sum: Float = 0
  for i in 0..<predicts.count {
    let difference = predicts[i] - expences[i]
    sum += pow(difference, 2)
  }
  return sum / Float(predicts.count)
}
