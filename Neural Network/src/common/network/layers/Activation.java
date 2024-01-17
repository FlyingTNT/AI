package common.network.layers;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

/**
 * An interface that represents an activation function for a {@link Layer}. Usually a {@link StandardLayer}
 * @author C. Cooper
 */
public interface Activation {
	/**
	 * Calculates the activation of the given matrix.
	 * @param values The values to take the activation of.
	 * @return The activation of the given values.
	 */
	SimpleMatrix activation(SimpleMatrix values);
	
	/**
	 * Calculates the derivative of this layer given a gradient matrix.
	 * <br><br>
	 * Generally not used because some activations, like {@link #SOFTMAX softmax}, also require inputs to calculate their derivatives.
	 * In this case, {@link #error(SimpleMatrix, SimpleMatrix) error(input, gradient)} is used.
	 * @param values A gradient.
	 * @return The derivative of this activation given the gradient.
	 */
	SimpleMatrix derivative(SimpleMatrix values);
	
	/**
	 * Given an input to this activation and the error coming out of the next layer, calculates the error of this activation.
	 * @param input The input to this activation that corresponds to the given error.
	 * @param nextWeightedError The error coming out of the next layer.
	 * @return The error of this activation.
	 */
	SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError);
	
	/**
	 * The name of this activation.
	 * @return The name of this activation.
	 */
	String name();
	
	/**
	 * Sigmoid activation.
	 */
	public static Activation SIGMOID = new Activation() {
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			return values.elementOp(new ElementOpReal() {//Applies the given function to each element in the given matrix.
				@Override
				public double op(int row, int col, double value) {
					return 1/(1+Math.exp(-value));//Applies the sigmoid function.
				}
			});
		}
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			return values.elementOp(new ElementOpReal() {//Calculates the sigmoid derivative for each position.
				@Override
				public double op(int row, int col, double value) {
					double val = 1/(1+Math.exp(-value));
					return val * (1-val);
				}
			});
		}
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {
			return input.elementOp(new ElementOpReal() {//Calculates the sigmoid derivative for each position.
				@Override
				public double op(int row, int col, double value) {
					double val = 1/(1+Math.exp(-value));
					return val * (1-val);
				}
			}).elementMult(nextWeightedError);
		}
		
		@Override
		public String name() {
			return "Sigmoid";
		}
	};
	
	/**
	 * The softmax activation. Each layer is output x depth. Takes softmax along the output axis.
	 */
	public static Activation SOFTMAX = new Activation() {
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {
			SimpleMatrix softmax = activation(input);
			SimpleMatrix error = softmax.elementMult(nextWeightedError);
			
			SimpleMatrix out = softmax.copy();
			for(int i = 0; i < input.getNumCols(); i++)
			{
				double sum = error.getColumn(i).elementSum();
				out.setColumn(i, softmax.getColumn(i).scale(-sum));
			}
			
			return out.plus(error);
		}
		
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			values = values.copy();//Copies values because this function modifies it.
			double[] maxes = new double[values.getNumCols()];//The maxes of each depth
			for(int i = 0; i < values.getNumCols(); i++)//Subtracts each column's max from each element in that column to give the function numeric stability. 
			{
				maxes[i] = values.getColumn(i).elementMax();
				values.setColumn(i, values.getColumn(i).minus(maxes[i]));
			}
			
			SimpleMatrix exp = values.elementExp();//e^value for each value in values
			
			/*
			 * For each column in exp, if its max isn't infinite, divides that column by the sum of the elements in that column.
			 * If it is infinite, sets each item in the column to 1/{# of rows}. This is done because when masking is applied, 
			 * all of the values could be -Infinity, which would cause an error.
			 */
			for(int d = 0; d < values.getNumCols(); d++)
				exp.setColumn(d, Double.isFinite(maxes[d]) ? exp.getColumn(d).divide(exp.getColumn(d).elementSum()) : new SimpleMatrix(1, values.getNumRows()));
			return exp;
		}
		
		@Override
		public String name() {
			return "Softmax";
		}
	};
	
	/**
	 * The softmax activation. Each layer is output x depth. Takes softmax along the depth axis.
	 */
	public static Activation SOFTMAX_DEPTHWISE = new Activation() {
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {			
			SimpleMatrix softmax = activation(input);
			SimpleMatrix error = softmax.elementMult(nextWeightedError);
			
			SimpleMatrix out = softmax.copy();
			for(int i = 0; i < input.getNumRows(); i++)
			{
				double sum = error.getRow(i).elementSum();
				out.setRow(i, softmax.getRow(i).scale(-sum));
			}
			
			return out.plus(error);
		}
		
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			values = values.copy();//Copies values because this function modifies it.
			double[] maxes = new double[values.getNumRows()];//The max of each row
			for(int i = 0; i < values.getNumRows(); i++)//Subtracts each row's max from each element in that row to give the function numeric stability. 
			{
				maxes[i] = values.getRow(i).elementMax();
				values.setRow(i, values.getRow(i).minus(maxes[i]));
			}
				
			
			SimpleMatrix exps = values.elementExp();//e^value for each value
			
			/*
			 * For each row in exp, if its max isn't infinite, divides that row by the sum of the elements in that row.
			 * If it is infinite, sets each item in the row to 1/{# of columns}. This is done because when masking is applied, 
			 * all of the values could be -Infinity, which would cause an error.
			 */
			for(int i = 0; i < values.getNumRows(); i++)
				exps.setRow(i, Double.isFinite(maxes[i]) ? exps.getRow(i).divide(exps.getRow(i).elementSum()) : SimpleMatrix.filled(1, values.getNumCols(), 1d/values.getNumCols()));
			return exps;
		}
		
		@Override
		public String name() {
			return "Softmax_Depthwise";
		}
	};
	
	/**
	 * No activation.
	 */
	public static Activation NONE = new Activation() {
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {
			return nextWeightedError;
		}
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			return SimpleMatrix.ones(values.getNumRows(), values.getNumCols());
		}
		
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			return values;
		}
		
		@Override
		public String name() {
			return "None";
		}
	};
	
	/**
	 * Rectified Linear Unit activation.
	 */
	public static Activation RELU = new Activation() {
		
		@Override
		public String name() {
			return "ReLU";
		}
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {
			return nextWeightedError.elementOp(new ElementOpReal() {
				@Override
				public double op(int row, int col, double value) {
					return input.get(row, col) > 0 ? value : 0;
				}
			});
		}
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
				return values.elementOp(new ElementOpReal() {
				@Override
				public double op(int row, int col, double value) {
					return value > 0 ? value : 0;
				}
			});
		}
	};
}
