package common.network.math;
import java.util.ArrayList;
import java.util.Iterator;

//import common.network.NetworkMatricies;
//import common.network.TrainingSet;

public class NetworkMath 
{
	public static float sigmoid(float value)
	{
		return (float) (1/(1+Math.exp(-value)));
	}
	
	public static float sigmoidPrime(float value)
	{
		float sigmoidValue = sigmoid(value);
		return sigmoidValue * (1 - sigmoidValue);
	}
	
	public static float[] sigmoidPrime(float[] value)
	{
		float[] output = new float[value.length];
		for(int i = 0; i < output.length; i++)
		{
			output[i] = sigmoidPrime(value[i]);
		}
		
		return output;
	}
	
	public static float dot(float[] a, float[] b)
	{
		if(a.length != b.length)
		{
			return Float.NaN;
		}
		float output = 0;
		for(int i = 0; i < a.length; i++)
		{
			output += a[i] * b[i];
		}
		return output;
	}
	
	public static float dot(float[] a)
	{
		float output = 0;
		for(int i = 0; i < a.length; i++)
		{
			output += a[i] * a[i];
		}
		return output;
	}
	
	public static int argmax(float[] vector)
	{
		float max = vector[0];
		int index = 0;
		for(int i = 1; i < vector.length; i++)
		{
			if(vector[i] > max)
			{
				index = i;
				max = vector[i];
			}
		}
		return index;
	}
	
	public static float normL2(float[][] vector)
	{
		float out = 0;
		for(int i = 0; i < vector.length; i++)
		{
			for(int j = 0; j < vector[0].length; j++)
			{
				out += vector[i][j] * vector[i][j];
			}
		}
		
		return (float)Math.sqrt(out);
	}
	
	public static float[][] sum(float[][]... matrixes)
	{
		float[][] out = new float[matrixes[0].length][matrixes[0][0].length];
		
		for(int i = 0; i < matrixes[0].length; i++)
		{
			for(int j = 0; j < matrixes[0][0].length; j++)
			{
				for(int k = 0; k < matrixes.length; k++)
				{
					out[i][j] += matrixes[k][i][j];
				}
			}
		}
		
		return out;
	}
	
	public static float[][] add(float[][] a, float[][] b)
	{
		float[][] out = new float[a.length][a[0].length];
		
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < a[0].length; j++)
			{
				out[i][j] = a[i][j] + b[i][j];
			}
		}
		
		return out;
	}
	
	public static float[][] subtract(float[][] a, float[][] b)
	{
		float[][] out = new float[a.length][a[0].length];
		
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < a[0].length; j++)
			{
				out[i][j] = a[i][j] - b[i][j];
			}
		}
		
		return out;
	}
	
	public static float[] elementwiseProd(float[] a, float[] b)
	{
		if(a.length != b.length)
		{
			System.out.println("Length mismatch: " + a.length + ", " + b.length);
			return new float[0];
		}
		float[] output = new float[a.length];
		for(int i = 0; i < a.length; i++)
		{
			output[i] = a[i] * b[i];
		}
		return output;
	}
	
	public static float[] hadamard(float[] a, float[] b)
	{
		return elementwiseProd(a, b);
	}
	
	public static float length(float[] a)
	{
		return (float) Math.sqrt(dot(a, a));
	}
	
	/**
	 * a-b
	 * @return a-b
	 */
	public static float[] subtract(float[] a, float[] b)
	{
		if(a.length != b.length)
		{
			return new float[0];
		}
		float[] output = new float[a.length];
		for(int i = 0; i < a.length; i++)
		{
			output[i] = a[i] - b[i];
		}
		return output;
	}
	
	public static float[] multiply(float[] a, float c)
	{
		float[] output = new float[a.length];
		for(int i = 0; i < a.length; i++)
		{
			output[i] = a[i] * c;
		}
		return output;
	}
	
	public static float[][] scale(float[][] a, float c)
	{
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < a[0].length; j++)
			{
				a[i][j] *= c;
			}
		}
		return a;
	}	
	
	public static float[][] multiplyAB(float[][] a, float[][] b)
	{
		float[][] output = new float[a.length][b[0].length];
		if(a[0].length != b.length)
		{
			throw new IllegalArgumentException("The matrixes are the wrong size, dumbass");
		}
		
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < b[0].length; j++)
			{
				for(int k = 0; k < a[0].length; k++)
				{
					output[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return output;
	}
	
	public static float[][] multiplyABT(float[][] a, float[][] b)
	{
		float[][] output = new float[a.length][b.length];
		if(a[0].length != b[0].length)
		{
			throw new IllegalArgumentException("The matrixes are the wrong size, dumbass");
		}
		
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < b.length; j++)
			{
				for(int k = 0; k < a[0].length; k++)
				{
					output[i][j] += a[i][k] * b[j][k];
				}
			}
		}
		return output;
	}
	
	public static float[][] multiplyATB(float[][] a, float[][] b)
	{
		float[][] output = new float[a[0].length][b[0].length];
		if(a.length != b.length)
		{
			throw new IllegalArgumentException("The matrixes are the wrong size, dumbass");
		}
		
		for(int i = 0; i < a[0].length; i++)
		{
			for(int j = 0; j < b[0].length; j++)
			{
				for(int k = 0; k < a.length; k++)
				{
					output[i][j] += a[k][i] * b[k][j];
				}
			}
		}
		return output;
	}
	
	public static float[][] multiplyATBT(float[][] a, float[][] b)
	{
		float[][] output = new float[a[0].length][b.length];
		if(a.length != b[0].length)
		{
			throw new IllegalArgumentException("The matrixes are the wrong size, dumbass");
		}
		
		for(int i = 0; i < a[0].length; i++)
		{
			for(int j = 0; j < b.length; j++)
			{
				for(int k = 0; k < a.length; k++)
				{
					output[i][j] += a[k][i] * b[j][k];
				}
			}
		}
		return output;
	}
	
	public static float[][] transpose(float[][] matrix)
	{
		float[][] output = new float[matrix[0].length][matrix.length];
		for(int i = 0; i < matrix.length; i++)
		{
			for(int j = 0; j < matrix[0].length; j++)
			{
				output[j][i] = matrix[i][j];
			}
		}
		return output;
	}
	
	public static float avg(float[] input)
	{
		float output = 0;
		for(float value : input)
		{
			output += value;
		}
		return output/input.length;
	}
	
	public static int binaryVectorToInt(int[] vector)
	{
		int output = 0;
		for(int i = 0; i < vector.length; i++)
		{
			output += vector[i] > 0 ? Math.pow(2, i) : 0;
		}
		return output;
	}
	
	public static int[] eintToBinaryVector(int value)
	{
		int i = 0;
		int value1 = value;
		
		for(i = 0; i >= 0; i++)
		{
			if(Math.pow(2, i) > value)
			{
				break;
			}
		}
		
		int[] output = value == 0 ? new int[]{0} : new int[i];
		i--;
		
		while(value1 > 0)
		{
			if(Math.pow(2, i) > value1)
			{
				output[i] = 0;
			}else {
				output[i] = 1;
				value1 -= Math.pow(2, i);
			}
			i--;
		}
		int[] outputCopy = new int[output.length];
		
		for(int j = 0; j < output.length; j++)
		{
			outputCopy[j] = output[output.length - (j + 1)];
		}
		
		return outputCopy;
	}
	
	public static float[] softmax(float[] weightedInputs)
	{
		float sum = 0;
		for(float input : weightedInputs)
		{
			sum += Math.exp(input);
		}
		float[] output = new float[weightedInputs.length];
		for(int i = 0; i < weightedInputs.length; i++)
		{
			output[i] = (float) (Math.exp(weightedInputs[i]) / sum);
		}
		return output;
	}
	/*
	public static enum costFunction
	{
		QUADRIATC, CROSS_ENTROPY
	}
	
	public static class quadraticCost
	{
		public static float cost(NetworkMatricies network, TrainingSet set)
		{
			float output = 0;
			
			for(float[][] ioPair : set.set())
			{
				output += dot(subtract(network.output(ioPair[0]), ioPair[1]));
			}
			
			return output / set.length();
		}
		
		public static float[] outputError(float[] output, float[] expected, float[] weightedInputs)
		{			

			
			return hadamard(subtract(output, expected), sigmoidPrime(weightedInputs));
		}
		
		public static float[] error(float[][] nextLayerWeights, float[] nextError, float[] weightedInputs)
		{

			
			return hadamard(transpose(multiplyATB(nextLayerWeights, new float[][]{nextError}))[0], sigmoidPrime(weightedInputs));
		}
		
		public static float[][] error(float[] output, float[] expected, float[][] weightedInputs, float[][][] weights)
		{//WeightedInputs should not have values for the nodes in the input row, but weights will
			ArrayList<float[]> errors = new ArrayList<>();
			errors.add(outputError(output, expected, weightedInputs[weightedInputs.length - 1]));
			
			for(int i = weightedInputs.length - 2; i > -1; i--)
			{
				errors.add(error(weights[i + 2], errors.get(errors.size() - 1), weightedInputs[i]));
			}
			
			ArrayList<float[]> outputList = new ArrayList<>();
			for(int i = errors.size() - 1; i > -1; i--)
			{
				outputList.add(errors.get(i));
			}
			
			return outputList.toArray(new float[0][0]);
		}*/
		
		/**
		 * @param valueIn The value that this weight's corresponding neuron gave to it
		 * @param neuronError The value of the error of the neuron this weight feeds data to
		 * @return The partial derivative of the cost with respect to this weight
		 */
	/*	public static float dCostdWeight(float valueIn, float neuronError)
		{
			return valueIn * neuronError;
		}
		
		public static float[][][] gradient(float[] output, float[] expected, float[][] weightedInputs, float[][][] weights, float[][] activations)
		{
			float[][] error = error(output, expected, weightedInputs, weights);
			
			return gradient(error, weights, activations);
		}
		
		public static float[][][] gradient(float[][] errors, float[][][] weights, float[][] activations)
		{
			float[][] error = errors;
			ArrayList<float[][]> gradient = new ArrayList<>();
			for(int i = 1; i < weights.length; i++)
			{
				ArrayList<float[]> column = new ArrayList<>();
				
				for(int j = 0; j < weights[i].length; j++)
				{
					float[] nodeGradients = new float[weights[i][j].length];
					for(int k = 0; k < nodeGradients.length; k++)
					{
						nodeGradients[k] = dCostdWeight(activations[i - 1][k], error[i - 1][j]);
					}
					column.add(nodeGradients);
				}
				
				gradient.add(column.toArray(new float[0][0]));
			}
			return gradient.toArray(new float[0][0][0]);
		}
		
	}

	public static class crossEntropyCost
	{
		public static float cost(NetworkMatricies network, TrainingSet set)
		{
			float output = 0;
			
			for(float[][] ioPair : set.set())
			{
				float[] value = network.output(ioPair[0]);
				for(int i = 0; i < set.outputs(); i++)
				{
					output += ioPair[1][i] * Math.log(value[i])
							+ (1 - ioPair[1][i]) * Math.log(1 - value[i]);
				}
			}
			
			return (float) (output * -1.0 / set.length());
		}
		
		public static float[] outputError(float[] output, float[] expected, float[] weightedInputs)
		{			
			return subtract(output, expected);
		}
		
		public static float[] error(float[][] nextLayerWeights, float[] nextError, float[] weightedInputs)
		{
			return quadraticCost.error(nextLayerWeights, nextError, weightedInputs);
		}
		
		public static float[][] error(float[] output, float[] expected, float[][] weightedInputs, float[][][] weights)
		{//WeightedInputs should not have values for the nodes in the input row, but weights will
			ArrayList<float[]> errors = new ArrayList<>();
			errors.add(outputError(output, expected, weightedInputs[weightedInputs.length - 1]));
			
			for(int i = weightedInputs.length - 2; i > -1; i--)
			{
				errors.add(error(weights[i + 2], errors.get(errors.size() - 1), weightedInputs[i]));
			}
			
			ArrayList<float[]> outputList = new ArrayList<>();
			for(int i = errors.size() - 1; i > -1; i--)
			{
				outputList.add(errors.get(i));
			}
			
			return outputList.toArray(new float[0][0]);
		}*/
		
		/**
		 * @param valueIn The value that this weight's corresponding neuron gave to it
		 * @param neuronError The value of the error of the neuron this weight feeds data to
		 * @return The partial derivative of the cost with respect to this weight
		 */
	/*	public static float dCostdWeight(float valueIn, float neuronError)
		{
			return quadraticCost.dCostdWeight(valueIn, neuronError);
		}
		
		public static float[][][] gradient(float[] output, float[] expected, float[][] weightedInputs, float[][][] weights, float[][] activations)
		{
			return quadraticCost.gradient(output, expected, weightedInputs, weights, activations);
		}
		
		public static float[][][] gradient(float[][] errors, float[][][] weights, float[][] activations)
		{
			return quadraticCost.gradient(errors, weights, activations);
		}

	}*/
}
