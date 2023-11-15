package common.network.layers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.layers.EmbeddingLayer;
import common.network.layers.layers.FlattenLayer;
import common.network.layers.layers.InputLayer;
import common.network.layers.layers.RotationLayer;
import common.network.layers.layers.StandardLayer;
import common.network.layers.models.LayersModel;

public class LayersMain {

	public static void main(String[] args) {
		LayersModel network;
		
		SimpleMatrix[][] training = new SimpleMatrix[256][2];
		
		for(int i = 0; i < 256; i++)
		{
			String binary = Integer.toBinaryString(i);
			float[][] out = new float[8][1];
			for(int j = 0; j < binary.length(); j++)
			{
				out[7 - j][0] = binary.charAt(binary.length() - 1 - j) == '0' ? 0 : 1;
			}
			training[i][0] = new SimpleMatrix(out);
			training[i][1] = new SimpleMatrix(out);
			//System.out.println(arrayToString(out));
		}
		
		InputLayer inputLayer = new InputLayer(8);
		StandardLayer outputLayer = new StandardLayer(inputLayer, 8, Activation.SIGMOID);
		network = new LayersModel(0.05f, Cost.QUADRATIC, inputLayer, outputLayer);
		
		network.feedForward(new SimpleMatrix(new float[][]{{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}})).print();
		
		System.out.println(network);
		
		for(int i = 0; i < 10; i++)
		{
			double cost = network.epoch(training);
			//System.out.println("============================================");
			System.out.println("Epoch " + i + ": " + cost);
		}
		
		System.out.println("============================================");
		for(int i = 0; i < 256; i++)
		{
			network.feedForward(training[i][0]);
		}
		
		SimpleMatrix[][] softTraining = new SimpleMatrix[4][2];
		
		for(int i = 0; i < 4; i++)
		{
			softTraining[i][0] = SimpleMatrix.filled(1, 1, i);
			softTraining[i][1] = SimpleMatrix.filled(1, 1, i);
		}
												  
        LayersModel softModel;
        InputLayer softIn = new InputLayer(1);
        EmbeddingLayer softMid = new EmbeddingLayer(softIn, 4, 4, false);
        //RotationLayer softFlat = new RotationLayer(softMid);
        StandardLayer softOut = new StandardLayer(softMid, 1, Activation.SOFTMAX_DEPTHWISE);
        softModel = new LayersModel(0.10f, Cost.SPARSE_CATEGORICAL_CROSS_ENTROPY, softIn, softMid, softOut);
        
        System.out.println(softModel);
		
		for(int i = 0; i < 20; i++)
		{
			double cost = softModel.epoch(softTraining);
			System.out.println("Epoch " + i + ": " + cost);
		}
		
		System.out.println("============================================");
		for(int i = 0; i < 4; i++)
		{
			softModel.feedForward(softTraining[i][0]);
		}
		
		File out = new File("C:\\AIClub\\Test\\linear.txt");
		try {
			FileWriter writer = new FileWriter(out);
			writer.write(softModel.stringify());
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
	
	
	public static String arrayToString(float[] array, int pointDigits)
	{
		DecimalFormat format = new DecimalFormat("0." + new String(new char[pointDigits]).replace("\0", "0"));
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
		DecimalFormat format = new DecimalFormat("0." + new String(new char[pointDigits]).replace("\0", "0"));
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
