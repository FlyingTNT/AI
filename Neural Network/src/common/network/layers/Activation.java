package common.network.layers;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

public interface Activation {
	
	SimpleMatrix activation(SimpleMatrix values);
	SimpleMatrix derivative(SimpleMatrix values);
	SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError);
	String name();
	
	public static Activation SIGMOID = new Activation() {
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			return values.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return 1/(1+Math.exp(-value));
				}
			});
		}
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			return values.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					double val = 1/(1+Math.exp(-value));
					return val * (1-val);
				}
			});
		}
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {
			return input.elementOp(new ElementOpReal() {
				
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
	
	public static Activation SOFTMAX = new Activation() {
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override//I have confirmed this derivative
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {
			SimpleMatrix softmax = activation(input);
			SimpleMatrix error = softmax.elementMult(nextWeightedError);
			
			float[][] out = new float[input.getNumRows()][input.getNumCols()];
			for(int d = 0; d < input.getNumCols(); d++)
			for(int output = 0; output < input.getNumRows(); output++)
			{
				for(int in = 0; in < input.getNumRows(); in++)
				{
					//DERIVATIVE OF iTH OUTPUT WITH RESPECT TO jTH INPUT = Sig(i) * ((i==j?1:0) - Sig(j));
					if(in == output)//COMPUTING THE PARTIAL FOR EACH INPUT
					{
						out[in][d] += (1 - softmax.get(in, d)) * error.get(output, d);
					}else {
						out[in][d] += -softmax.get(in, d)* error.get(output, d);
					}
				}
			}
			
			return new SimpleMatrix(out);
		}
		
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			values = values.copy();
			double[] maxes = new double[values.getNumCols()]; 
			for(int i = 0; i < values.getNumCols(); i++)
			{
				maxes[i] = values.getColumn(i).elementMax();
				values.setColumn(i, values.getColumn(i).minus(maxes[i]));
			}
			
			SimpleMatrix exp = values.elementExp();
			
			for(int d = 0; d < values.getNumCols(); d++)
				exp.setColumn(d, Double.isFinite(maxes[d]) ? exp.getColumn(d).divide(exp.getColumn(d).elementSum()) : new SimpleMatrix(1, values.getNumRows()));
			return exp;
		}
		
		@Override
		public String name() {
			return "Softmax";
		}
	};
	
	//VERIFIED
	public static Activation SOFTMAX_DEPTHWISE = new Activation() {
		
		@Override
		public SimpleMatrix derivative(SimpleMatrix values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override
		public SimpleMatrix error(SimpleMatrix input, SimpleMatrix nextWeightedError) {			
			SimpleMatrix softmax = activation(input);
			SimpleMatrix error = softmax.elementMult(nextWeightedError);
			
			float[][] out = new float[input.getNumRows()][input.getNumCols()];
			for(int w = 0; w < input.getNumRows(); w++)
			for(int output = 0; output < input.getNumCols(); output++)
			{
				for(int in = 0; in < input.getNumCols(); in++)
				{
					//DERIVATIVE OF iTH OUTPUT WITH RESPECT TO jTH INPUT = Sig(i) * ((i==j?1:0) - Sig(j));
					if(in == output)//COMPUTING THE PARTIAL FOR EACH INPUT
					{
						out[w][in] += (1 - softmax.get(w, in)) * error.get(w, output);
					}else {
						out[w][in] += -softmax.get(w, in)* error.get(w, output);
					}
				}
			}
			
			return new SimpleMatrix(out);
		}
		
		@Override
		public SimpleMatrix activation(SimpleMatrix values) {
			values = values.copy();
			double[] maxes = new double[values.getNumRows()]; 
			for(int i = 0; i < values.getNumRows(); i++)
			{
				maxes[i] = values.getRow(i).elementMax();
				values.setRow(i, values.getRow(i).minus(maxes[i]));
			}
				
			
			SimpleMatrix exps = values.elementExp();
			for(int i = 0; i < values.getNumRows(); i++)
				exps.setRow(i, Double.isFinite(maxes[i]) ? exps.getRow(i).divide(exps.getRow(i).elementSum()) : SimpleMatrix.filled(1, values.getNumCols(), 0));
			return exps;
		}
		
		@Override
		public String name() {
			return "Softmax_Depthwise";
		}
	};
	
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
			return values.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return value > 0 ? 1 : 0;
				}
			});
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
