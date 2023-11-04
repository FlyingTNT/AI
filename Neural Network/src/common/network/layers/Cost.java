package common.network.layers;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

public interface Cost {
	double cost(SimpleMatrix prediction, SimpleMatrix target);
	SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target);
	
	public static Cost QUADRATIC = new Cost() {
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target) {
			return prediction.minus(target);
		}
		
		@Override
		public double cost(SimpleMatrix prediction, SimpleMatrix target) {
				double x = prediction.minus(target).normF();
				return x*x/2;
		}
	};
	
	//VERIFIED
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
	
	public static Cost SPARSE_CATEGORICAL_CROSS_ENTROPY = new Cost() {
		@Override//This seems wrong but I derived it myself and it is right.
		public SimpleMatrix derivative(SimpleMatrix prediction, SimpleMatrix target) {
				return prediction.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					int index = (int)target.get(row, 0);
					if(index == -1)
						return 0;
					return (value - ((index == col) ? 1 : 0)) / (value * (1-value));
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
	
	public static Cost SPARSE_CATEGORICAL_CROSS_ENTROPY_WIDTHWISE = new Cost() {
		@Override//This seems wrong but I derived it myself and it is right.
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
