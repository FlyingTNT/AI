package common.network.layers;

import common.network.math.NetworkMath;

public interface Cost {
	float cost(float[][] prediction, float[][] target);
	float[][] derivative(float[][] prediction, float[][] target);
	
	public static Cost QUADRATIC = new Cost() {
		
		@Override
		public float[][] derivative(float[][] prediction, float[][] target) {
			return NetworkMath.subtract(prediction, target);
		}
		
		@Override
		public float cost(float[][] prediction, float[][] target) {
				float x = NetworkMath.length(NetworkMath.subtract(target, prediction)[0]);
				return x*x/2;
		}
	};
	
	//VERIFIED
	public static Cost CROSS_ENTROPY = new Cost() {
		@Override
		public float[][] derivative(float[][] prediction, float[][] target) {
			float[][] out = new float[prediction.length][prediction[0].length];
			for(int i = 0; i < prediction.length; i++)
			{
				for(int j = 0; j < prediction[0].length; j++)
				{
					out[i][j] = (prediction[i][j] - target[i][j]) / (prediction[i][j] * (1 - prediction[i][j]));
				}
			}
			return out;
		}
		
		@Override
		public float cost(float[][] prediction, float[][] target) {
			float sum = 0;
			for(int i = 0; i < prediction.length; i++)
			{
				for(int j = 0; j < prediction[0].length; j++)
				{
					sum += target[i][j] * Math.log(prediction[i][j]);
				}
			}
			return -sum;
		}
	};
	
	public static Cost SPARSE_CATEGORICAL_CROSS_ENTROPY = new Cost() {
		@Override//This seems wrong but I derived it myself and it is right.
		public float[][] derivative(float[][] prediction, float[][] target) {
			float[][] out = new float[prediction.length][prediction[0].length];
			for(int i = 0; i < prediction.length; i++)
			{
				int goal = (int)target[i][0];
				//System.out.println(LayersMain.floatMatrixToString(prediction, 1));
				//System.out.println(goal);
				for(int j = 0; j < prediction[0].length; j++)
				{
					out[i][j] = (prediction[i][j] - (j == goal ? 1 : 0)) / ( prediction[i][j] * (1 -  prediction[i][j]));
				}
			}
			return out;
		}
		
		@Override
		public float cost(float[][] prediction, float[][] target) {
			float sum = 0;
			for(int i = 0; i < prediction.length; i++)
			{
				int goal = (int)target[i][0];
				for(int j = 0; j < prediction[0].length; j++)
				{
					sum += (j == goal ? 1 : 0) * Math.log(prediction[i][j]);
				}
			}
			return -sum;
		}
	};
}
