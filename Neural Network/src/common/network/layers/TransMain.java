package common.network.layers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.TransformerModel;

public class TransMain {	
	public static void main(String[] args) {		
		TransformerModel transformer = new TransformerModel(0.005f, 8, 12, 8, 3, 6);
		SimpleMatrix[][] transformerData = new SimpleMatrix[64][2];
		
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
		
		//transformer.test(transformerData);
		
		DecimalFormat format = new DecimalFormat("0.000");
		
		float cost = 100;
		
		int count = 0;
		
		for(int i = 0; count < 5; i++)
		{
			cost = transformer.epoch(transformerData);
			System.out.println("Epoch " + (i + 1) + ", Cost: " + format.format(cost));
			if(cost < 1)
				count++;
			else
				count = 0;
		}
		
		System.out.println("=======================================");
		
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
