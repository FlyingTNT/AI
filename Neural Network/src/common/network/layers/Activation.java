package common.network.layers;

import java.util.Arrays;

import common.network.math.NetworkMath;

public interface Activation {
	
	float[][] activation(float[][] values);
	float[][] derivative(float[][] values);
	float[][] error(float[][] input, float[][] nextWeightedError);
	String name();
	
	public static Activation SIGMOID = new Activation() {
		@Override
		public float[][] activation(float[][] values) {
			float[][] out = new float[values.length][values[0].length];
			for(int d = 0; d < values[0].length; d++)
			for(int i = 0; i < values.length; i++)
			{
				out[i][d] = NetworkMath.sigmoid(values[i][d]);
			}
			return out;
		}
		
		@Override
		public float[][] derivative(float[][] values) {
			float[][] out = new float[values.length][values[0].length];
			for(int i = 0; i < values.length; i++)
			{
				out[i] = NetworkMath.sigmoidPrime(values[i]);
			}
			return out;
		}
		
		@Override
		public float[][] error(float[][] input, float[][] nextWeightedError) {
			float[][] out = new float[input.length][input[0].length];
			for(int i = 0; i < input.length; i++)
			{
				out[i] = NetworkMath.hadamard(NetworkMath.sigmoidPrime(input[i]), nextWeightedError[i]);
			}
			return out;
		}
		
		@Override
		public String name() {
			return "Sigmoid";
		}
	};
	
	public static Activation SOFTMAX = new Activation() {
		
		@Override
		public float[][] derivative(float[][] values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override//I have confirmed this derivative
		public float[][] error(float[][] input, float[][] nextWeightedError) {
			float[][] softmax = activation(input);
			float[][] out = new float[input.length][input[0].length];
			for(int d = 0; d < input[0].length; d++)
			for(int output = 0; output < input.length; output++)
			{
				for(int in = 0; in < input.length; in++)
				{
					//DERIVATIVE OF iTH OUTPUT WITH RESPECT TO jTH INPUT = Sig(i) * ((i==j?1:0) - Sig(j));
					if(in == output)//COMPUTING THE PARTIAL FOR EACH INPUT
					{
						out[in][d] += softmax[output][d] * (1 - softmax[in][d]) * nextWeightedError[output][d];
					}else {
						out[in][d] += softmax[output][d] * -softmax[in][d]* nextWeightedError[output][d];
					}
				}
			}
			return out;
		}
		
		@Override
		public float[][] activation(float[][] values) {
			float[] sum = new float[values[0].length];
			float[][] exps = new float[values.length][values[0].length];
			
			for(int d = 0; d < values[0].length; d++)
			for(int i = 0; i < values.length; i++)
			{
				exps[i][d] = (float) Math.exp(values[i][d]);
				if(Float.isNaN(exps[i][d]))
				{
					System.out.println(LayersMain.floatMatrixToString(values, 2));
					System.out.println("(" + i + ", " + d + ")");
					throw new IllegalArgumentException("Exp on too large number!");
				}
				sum[d] += exps[i][d];
			}
			
			float[][] out = new float[values.length][values[0].length];
			for(int d = 0; d < values[0].length; d++)
			for(int i = 0; i < values.length; i++)
			{
				if(sum[d] == 0)
				{
					throw new IllegalStateException("Softmax sum is zero!");
				}else if(Double.isNaN(sum[d])) {
					throw new IllegalStateException("Softmax sum overflowed!");
				}
				out[i][d] = exps[i][d]/sum[d];
			}
			return out;
		}
		
		@Override
		public String name() {
			return "Softmax";
		}
	};
	
	//VERIFIED
	public static Activation SOFTMAX_DEPTHWISE = new Activation() {
		
		@Override
		public float[][] derivative(float[][] values) {
			throw new IllegalStateException("DOES NOT WORK DO NOT USE; USE ERROR");
		}
		
		@Override
		public float[][] error(float[][] input, float[][] nextWeightedError) {
			float[][] softmax = activation(input);
			float[][] out = new float[input.length][input[0].length];
			
			for(int i = 0; i < input.length; i++)
			{
				for(int j = 0; j < input[0].length; j++)
				{
					for(int k = 0; k < input[0].length; k++)
					{
						if(j == k)
						{
							out[i][j] += softmax[i][k] * (1 - softmax[i][k]) * nextWeightedError[i][k];
						}else {
							out[i][j] += softmax[i][j] * -softmax[i][k] * nextWeightedError[i][k];
						}
					}
				}
			}
			return out;
		}
		
		@Override
		public float[][] activation(float[][] values) {
			double[] sum = new double[values.length];
			double[][] exps = new double[values.length][values[0].length];
			
			for(int d = 0; d < values[0].length; d++)
			{
				for(int i = 0; i < values.length; i++)
				{
					exps[i][d] = Math.exp(values[i][d]);
					if(Double.isNaN(exps[i][d]))
					{
						System.out.println(LayersMain.floatMatrixToString(values, 2));
						System.out.println("(" + i + ", " + d + ")");
						throw new IllegalArgumentException("Exp on too large number!");
					}
					sum[i] += exps[i][d];
				}
			}
			
			/*for(int i = 0; i < values.length; i++)
			{
				if(Double.isInfinite(sum[i]))
				{
					sum[i] = Double.MAX_VALUE;
					for(int j = 0; j < values[0].length; j++)
					{
						if(Double.isInfinite(exps[i][j]))
						{
							exps[i][j] = Double.MAX_VALUE;
						}
					}
				}
			}*/
			
			float[][] out = new float[values.length][values[0].length];
			for(int d = 0; d < values[0].length; d++)
			for(int i = 0; i < values.length; i++)
			{
				//if(sum[i] == 0)
				//{
				//	sum[i] = values[0].length;
				//	for(int j = 0; j < values[0].length; j++)
				//	{
				//		exps[i][j] = 1;
				//	}
				//}else
				if(Double.isNaN(sum[i])) {
					throw new IllegalStateException("Softmax sum overflowed!");
				}
				out[i][d] = (float)(exps[i][d]/sum[i]);
			}
			return out;
		}
		
		@Override
		public String name() {
			return "Softmax Depthwise";
		}
	};
	
	public static Activation NONE = new Activation() {
		
		@Override
		public float[][] error(float[][] input, float[][] nextWeightedError) {
			return nextWeightedError;
		}
		
		@Override
		public float[][] derivative(float[][] values) {
			float[][] out = new float[values.length][values[0].length];
			float[] ones = new float[values[0].length];
			Arrays.fill(ones, 1);
			Arrays.fill(out, ones);
			return out;
		}
		
		@Override
		public float[][] activation(float[][] values) {
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
		public float[][] error(float[][] input, float[][] nextWeightedError) {
			float[][] out = new float[input.length][input[0].length];
			for(int d = 0; d < input[0].length; d++)
			for(int i = 0; i < input.length; i++)
			{
				out[i][d] = input[i][d] <= 0 ? 0 : nextWeightedError[i][d];
			}
			return out;
		}
		
		@Override
		public float[][] derivative(float[][] values) {
			float[][] out = new float[values.length][values[0].length];
			for(int d = 0; d < values[0].length; d++)
			for(int i = 0; i < values.length; i++)
			{
				out[i][d] = values[i][d] <= 0 ? 0 : 1;
			}
			return out;
		}
		
		@Override
		public float[][] activation(float[][] values) {
			float[][] out = new float[values.length][values[0].length];
			for(int d = 0; d < values[0].length; d++)
			for(int i = 0; i < values.length; i++)
			{
				out[i][d] = values[i][d] <= 0 ? 0 : values[i][d];
			}
			return out;
		}
	};
}
