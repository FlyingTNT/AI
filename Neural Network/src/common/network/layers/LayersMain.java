package common.network.layers;

import java.text.DecimalFormat;

import common.network.layers.layers.EmbeddingLayer;
import common.network.layers.layers.FlattenLayer;
import common.network.layers.layers.InputLayer;
import common.network.layers.layers.StandardLayer;
import common.network.layers.models.LayersNetwork;

public class LayersMain {

	public static void main(String[] args) {
		LayersNetwork network;
		
		float[][][][] training = new float[256][2][8][1];
		
		for(int i = 0; i < 256; i++)
		{
			String binary = Integer.toBinaryString(i);
			float[][] out = new float[8][1];
			for(int j = 0; j < binary.length(); j++)
			{
				out[7 - j][0] = binary.charAt(binary.length() - 1 - j) == '0' ? 0 : 1;
			}
			training[i][0] = out;
			training[i][1] = out;
			//System.out.println(arrayToString(out));
		}
		
		InputLayer inputLayer = new InputLayer(8);
		StandardLayer outputLayer = new StandardLayer(inputLayer, 8, Activation.SIGMOID);
		network = new LayersNetwork(0.05f, Cost.QUADRATIC, inputLayer, outputLayer);
		
		System.out.println(floatMatrixToString(network.feedForward(new float[][]{{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}}), 2));
		
		System.out.println(network);
		
		for(int i = 0; i < 10; i++)
		{
			network.epoch(training);
			System.out.println("============================================");
			System.out.println(network);
		}
		
		System.out.println("============================================");
		for(int i = 0; i < 256; i++)
		{
			System.out.println(floatMatrixToString(network.feedForward(training[i][0]), 2));
		}
		
		float[][][][] softTraining = new float[][][][]{{{{0}}, {{1}, {0}, {0}, {0}}},
												       {{{1}}, {{0}, {1}, {0}, {0}}},
												       {{{2}}, {{0}, {0}, {1}, {0}}},
												       {{{3}}, {{0}, {0}, {0}, {1}}}};
												  
        LayersNetwork softModel;
        InputLayer softIn = new InputLayer(1);
        EmbeddingLayer softMid = new EmbeddingLayer(softIn, 4, 4, false);
        FlattenLayer softFlat = new FlattenLayer(softMid);
        StandardLayer softOut = new StandardLayer(softFlat, 4, Activation.SOFTMAX);
        softModel = new LayersNetwork(0.10f, Cost.CROSS_ENTROPY, softIn, softMid, softFlat, softOut);
        
        System.out.println(softModel);
		
		for(int i = 0; i < 20; i++)
		{
			softModel.epoch(softTraining);
			System.out.println("============================================");
			System.out.println(softModel);
		}
		
		System.out.println("============================================");
		for(int i = 0; i < 4; i++)
		{
			System.out.println(floatMatrixToString(softModel.feedForward(softTraining[i][0]), 2));
		}
	}
	
	
	public static String arrayToString(float[] array, int pointDigits)
	{
		DecimalFormat format = new DecimalFormat("0." + "0".repeat(pointDigits));
		if(array.length == 0)//If the array is empty, return {}.
		{
			return "{}";
		}
		String output = "{";//Open the output string with "{"
		for(int i = 0; i < array.length - 1; i++)
		{
			output += format.format(array[i]) + ", ";//For each item but the last one, add the item and ", " to the array.
		}
		output += format.format(array[array.length - 1]) + "}";//Caps the string with the last item and "}"
		return output;
	}

	public static String floatMatrixToString(float[][] matrix, int pointDigits)
	{
		DecimalFormat format = new DecimalFormat("0." + "0".repeat(pointDigits));
		String output = "";//Initializes an output string.
		for(int i = 0; i < matrix.length; i++)//For row in the matrix.
		{
			output = output + "[";//Start the row with an open bracket
			for(int j = 0; j < matrix[i].length; j++) //For column in the row
			{
				/*
				 * Add 0 if the value is false and 1 if true. If this column isn't the last, also add ", "
				 */
				if(j != matrix[i].length - 1)
				{
					output = output + format.format(matrix[i][j]) + ", ";
				}else {
					output = output +  format.format(matrix[i][j]);
				}
			}
			output = output + "]\n";//Cap the row with a closing bracket and a newline.
		}
		return output;//Return the output String.
	}
	
	public static void print(float[][] matrix)
	{
		System.out.println(floatMatrixToString(matrix, 2));
	}
}
