package common.network.math;
import org.ejml.simple.SimpleMatrix;

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
	
	public static double[][] scale(double[][] a, double c)
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
	
	public static double[][] multiplyAB(double[][] a, double[][] b)
	{
		double[][] output = new double[a.length][b[0].length];
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
	
	public static double[][] multiplyABT(double[][] a, double[][] b)
	{
		double[][] output = new double[a.length][b.length];
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
	
	public static int argmax(SimpleMatrix matrix)
	{
		int index = -1;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < matrix.getNumCols(); i++)
			if(matrix.get(i) > max)
			{
				max = matrix.get(i);
				index = i;
			}
		return index;
	}
}
