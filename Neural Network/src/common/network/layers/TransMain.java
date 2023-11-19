package common.network.layers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.TransformerModel;

<<<<<<< HEAD
public class TransMain {

	public static void main(String[] args) {		
		TransformerModel transformer = new TransformerModel(0.005f, 8, 8, 8, 4, 6);
		float[][][][] transformerData = new float[64][2][8][1];
=======
/**
 * Main class that tests a transformer model on the dataset:<br>
 * 1 1 1 1 1 1 1 1 -> 1 1 1 1 1 1 1 1<br>
 * 1 2 1 2 1 2 1 2 -> 2 1 2 1 2 1 2 1<br>
 * 1 3 1 3 1 3 1 3 -> 3 1 3 1 3 1 3 1<br>
 * ....<br>
 * 8 7 8 7 8 7 8 7 -> 7 8 7 8 7 8 7 8<br>
 * 8 8 8 8 8 8 8 8 -> 8 8 8 8 8 8 8 8
 * @author C. Cooper
 */
public class TransMain {	
	public static void main(String[] args) {	
		TransformerModel transformer = new TransformerModel(0.005f, 8, 8, 12, 9, 8, 3, 6);
		SimpleMatrix[][] transformerData = new SimpleMatrix[64][2];
>>>>>>> refs/remotes/origin/ejml
		
		/*
		 * Generating dataset
		 */
		int pos = 0;
		for(int i = 0; i < 8; i++)
		{
			for(int j = 0; j < 8; j++)
			{
				SimpleMatrix in = SimpleMatrix.filled(8, 1, 0);
				SimpleMatrix out = SimpleMatrix.filled(8, 1, 0);
				for(int k = 0; k < 8; k++)
				{
					if(k % 2 == 0)
					{
						in.set(k, 0, i+1);
						out.set(k, 0, j+1);
					}else {
						
						in.set(k, 0, j+1);
						out.set(k, 0, i+1);
					}
				}
				transformerData[pos][0] = in;
				transformerData[pos][1] = out;
				pos++;
			}
		}
		System.out.println("=======================================");
		
		System.out.println(transformer);
		
		DecimalFormat format = new DecimalFormat("0.000");
		
		float cost = 100;
		
<<<<<<< HEAD
		for(int i = 0; i < 200; i++)
=======
		int count = 0;
		
		/*
		 * Runs epochs until the cost has been less than 1 for 5 epochs in a row.
		 * I found that this produced the best results. If you go for like 10 in a row,
		 * the model gets worse.
		 */
		for(int i = 0; count < 5; i++)
>>>>>>> refs/remotes/origin/ejml
		{
			cost = transformer.epoch(transformerData);
			System.out.println("Epoch " + (i + 1) + ", Cost: " + format.format(cost));
			if(cost < 1)
				count++;
			else
				count = 0;
		}
		
		System.out.println("=======================================");
		
		/*
		 * For each point in the dataset, prints out input -> actualInferenceOutput ~ target
		 */
		for(int i = 0; i < 64; i++)
		{
			SimpleMatrix result = transformer.beamSearch(transformerData[i][0], 10);
			for(int j = 0; j < 8; j++)
			{
				System.out.println(transformerData[i][0].get(j, 0) + " -> " + result.get(j+1, 0) + " ~ " + transformerData[i][1].get(j, 0));
			}
			System.out.println();
		}
		
		//transformer.test(transformerData);
		
		/*
		 * Store the model so it can be reloaded in LoadMain
		 */
		File out = new File("C:\\AIClub\\Test\\model.txt");
		try {
			FileWriter writer = new FileWriter(out);
			writer.write(transformer.stringify());
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
}
