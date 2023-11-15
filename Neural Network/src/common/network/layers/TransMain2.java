package common.network.layers;

import java.text.DecimalFormat;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.TransformerModel;

public class TransMain2 {

	public static void main(String[] args) {
		TransformerModel transformer = new TransformerModel(0.05f, 9, 9, 16, 9, 9, 4, 1);
		SimpleMatrix[][] transformerData = new SimpleMatrix[][] {
			{new SimpleMatrix(new float[][]{{1}, {4}, {7}, {4}, {8}, {5}, {2}, {3}, {1}}), new SimpleMatrix(new float[][]{{1}, {3}, {7}, {4}, {4}, {5}, {8}, {1}, {6}})},
			{new SimpleMatrix(new float[][]{{4}, {8}, {3}, {8}, {7}, {5}, {1}, {2}, {5}}), new SimpleMatrix(new float[][]{{7}, {4}, {8}, {8}, {1}, {6}, {5}, {3}, {2}})}
		};
		
		System.out.println(transformer);
		
		DecimalFormat format = new DecimalFormat("0.000");
		
		float cost = 100;
		
		for(int i = 0; i < 100; i++)
		{
			cost = transformer.epoch(transformerData);
			System.out.println("Epoch " + (i + 1) + ", Cost: " + format.format(cost));
		}
		//transformer.epoch(new float[][][][] { transformerData[1]});
		
		System.out.println("=======================================");
		
		for(int i = 0; i < transformerData.length; i++)
		{
			SimpleMatrix result = transformer.beamSearch(transformerData[i][0], 4);
			for(int j = 0; j < 9; j++)
			{
				System.out.println(transformerData[i][0].get(j, 0) + " -> " + result.get(j+1, 0));
			}
			//System.out.println();
			//LayersMain.print(result);
			System.out.println();
			transformerData[i][1].print();
			System.out.println();
		}
		
		System.out.println("Beam:");
		
		transformer.beamSearch(transformerData[0][0], 4).print();
		
		System.out.println();
		
		transformerData[0][1].print();
		
		
		transformer.test(transformerData);
		
		transformer.feedForward(transformerData[0][0]);
	}

}
