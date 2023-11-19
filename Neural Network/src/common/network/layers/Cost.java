package common.network.layers;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

/**
 * An interface that represents a cost (or loss) function of a machine learning model.
 * I know cost and loss don't mean the same thing and this should really be called Loss. I don't care.
 * @author C. Cooper
 */
public interface Cost {
	/**
	 * Calculates the loss for a given prediction and target.
	 * @param prediction The prediction.
	 * @param target The target.
	 * @return The loss of the prediction given the target.
	 */
	double cost(SimpleMatrix prediction, SimpleMatrix target);
	
	/**
	 * Calculates the derivative of the loss function given a prediction and target.
	 * @param prediction The prediction.
	 * @param target The target.
	 * @return The derivative of the loss function given a prediction and target.
	 */
	SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target);
	
	/**
	 * Quadratic loss.<br>
	 * Loss = sqrt(sum(prediction - target)^2) / 2
	 */
	public static Cost QUADRATIC = new Cost() {
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target) {
			return prediction.minus(target);
		}
		
		@Override
		public double cost(SimpleMatrix prediction, SimpleMatrix target) {
				double x = prediction.minus(target).normF();//Magnitude of prediction - target
				return x*x/2;
		}
	};
	
	/**
	 * Cross entropy loss.<br>
	 * Loss = -sum(ln(prediction) * target)
	 */
	public static Cost CROSS_ENTROPY = new Cost() {
		@Override
		public SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target) {
			return prediction.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return (value - target.get(row, col)) / (value * (1-value));
				}
			});
		}
		
		@Override
		public double cost(SimpleMatrix prediction, SimpleMatrix target) {
			return -target.elementMult(prediction.elementLog()).elementSum();
		}
	};
	
	/**
	 * {@link #CROSS_ENTROPY Cross Entropy} loss except the targets are given as the integer indexes of the targets, rather than the whole
	 * one-hot encoding.
	 * <br><br>
	 * ex: Target is just 3 instead of {0, 0, 0, 1, 0}
	 */
	public static Cost SPARSE_CATEGORICAL_CROSS_ENTROPY = new Cost() {
		@Override
		public SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target) {
				return prediction.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					int index = (int)target.get(row, 0);
					if(index == -1)
						return 0;
					double out = (value - ((index == col) ? 1 : 0)) / (value * (1-value));
					return out > 5 ? 5 : out < -5 ? -5 : out;//Clips the gradients if they're over 5 in magnitude
					//return -((index == col) ? 1 : 0)/value;
				}
			});
		}
		
		@Override
		public double cost(SimpleMatrix prediction, SimpleMatrix target) {
			double sum = 0;
			for(int i = 0; i < target.getNumRows(); i++)
			{
				if((int)target.get(i, 0) == -1)
					continue;
				sum += -Math.log(prediction.get(i, (int)target.get(i, 0)));
			}
			return sum;
		}
	};
	
	/**
	 * {@link #SPARSE_CATEGORICAL_CROSS_ENTROPY SparseCategoricalCrossEntropy} loss but along the outputSize axis, rather than the
	 * depth axis. ({@link #SPARSE_CATEGORICAL_CROSS_ENTROPY SparseCategoricalCrossEntropy} is made for {@link Activation#SOFTMAX_DEPTHWISE depthwise softmax},
	 * and this is made for {@link Activation#SOFTMAX normal softmax}).
	 */
	public static Cost SPARSE_CATEGORICAL_CROSS_ENTROPY_WIDTHWISE = new Cost() {
		@Override
		public SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target) {
				return prediction.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					int index = (int)target.get(col, 0);
					if(index == -1)
						return 0;
					return (value - ((index == row) ? 1 : 0)) / (value * (1-value));
				}
			});
		}
		
		@Override
		public double cost(SimpleMatrix prediction, SimpleMatrix target) {
			double sum = 0;
			for(int i = 0; i < target.getNumRows(); i++)
			{
				if((int)target.get(i, 0) == -1)
					continue;
				sum += -Math.log(prediction.get((int)target.get(i, 0), i));
			}
			return sum;
		}
	};
}
