public class MSE {
  public class func MSE(predicts: [Float], expences: [Float]) -> Float {
    var sum: Float = 0
    for i in 0..<predicts.count {
      let difference = predicts[i] - expences[i]
      let squared_difference = difference * difference
      sum += squared_difference
    }
    return sum / Float(predicts.count)
  }
}
