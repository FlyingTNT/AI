package common.network.layers;

import java.text.DecimalFormat;

import common.network.layers.layers.AttentionLayer;
import common.network.layers.layers.EmbeddingLayer;
import common.network.layers.layers.FlattenLayer;
import common.network.layers.layers.InputLayer;
import common.network.layers.layers.ResidualAddition;
import common.network.layers.layers.StandardLayer;
import common.network.layers.models.LayersNetwork;
import common.network.layers.models.TransformerModel;
import common.network.math.NetworkMath;

public class TransMain {

	public static void main(String[] args) {		
		TransformerModel transformer = new TransformerModel(0.005f, 8, 8, 8, 4, 6);
		float[][][][] transformerData = new float[64][2][8][1];
		
		int pos = 0;
		for(int i = 0; i < 8; i++)
		{
			for(int j = 0; j < 8; j++)
			{
				float[][] in = new float[8][1];
				float[][] out = new float[8][1];
				for(int k = 0; k < 8; k++)
				{
					if(k % 2 == 0)
					{
						in[k][0] = i+1;
						out[k][0] = j+1;
					}else {
						in[k][0] = j+1;
						out[k][0] = i+1;
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
		
		for(int i = 0; i < 200; i++)
		{
			cost = transformer.epoch(transformerData);
			System.out.println("Epoch " + (i + 1) + ", Cost: " + format.format(cost));
		}
		//transformer.epoch(new float[][][][] { transformerData[1]});
		
		System.out.println("=======================================");
		
		for(int i = 0; i < 64; i++)
		{
			float[][] result = transformer.beamSearch(transformerData[i][0], 9);
			for(int j = 0; j < 8; j++)
			{
				System.out.println(transformerData[i][0][j][0] + " -> " + result[j+1][0] + " ~ " + transformerData[i][1][j][0]);
			}
			System.out.println();
		}
	}

}
