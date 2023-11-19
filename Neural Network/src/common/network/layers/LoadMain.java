package common.network.layers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;
import common.network.layers.models.LayersModel;
import common.network.layers.models.TransformerModel;

/**
 * Main class that loads the models from {@link LayersMain} and {@link TransMain} and runs inference on them.
 * Used to verify that the load functionality works, and to test internal layer updates without having to train
 * every time.
 * @author C. Cooper
 */
public class LoadMain {

	public static void main(String[] args) {
		/*
		 * LayersMain model
		 */
		String string = null;
		try {
			string = Files.readString(Paths.get("C:\\AIClub\\Test\\linear.txt"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		LayersModel model = LayersModel.load(string);
		
		SimpleMatrix[][] softTraining = new SimpleMatrix[4][2];
		
		for(int i = 0; i < 4; i++)
		{
			softTraining[i][0] = SimpleMatrix.filled(1, 1, i);
			softTraining[i][1] = SimpleMatrix.filled(1, 1, i);
		}
		
		System.out.println("============================================");
		for(int i = 0; i < 4; i++)
		{
			System.out.println(i);
			model.feedForward(softTraining[i][0]).print();
		}
		
		string = null;
		try {
			string = Files.readString(Paths.get("C:\\AIClub\\Test\\model.txt"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		/*
		 * TransMain model
		 */
		
		TransformerModel transformer = TransformerModel.load(string);
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
		
		System.out.println("=======================================");
		
		double costSum = 0;
		
		for(int i = 0; i < 64; i++)
		{
			SimpleMatrix result = transformer.beamSearch(transformerData[i][0], 10);
			//result.print();
			//result.rows(1, result.getNumRows()).print();
			SimpleMatrix out = new SimpleMatrix(8, 9).elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return col == ((int)result.get(row+1, 0)) ? 0.99 : 0.01;
				}
			});
			//out.print();
			//transformerData[i][1].print();
			double cost = Cost.SPARSE_CATEGORICAL_CROSS_ENTROPY.cost(out, transformerData[i][1]);
			costSum += cost;
			for(int j = 0; j < 8; j++)
			{
				System.out.println(transformerData[i][0].get(j, 0) + " -> " + result.get(j+1, 0) + " ~ " + transformerData[i][1].get(j, 0));
			}
			System.out.println("Cost: " + cost);
			System.out.println();
		}
		
		System.out.println("Cost: " + costSum / transformerData.length);
		
		//transformer.test(transformerData);
	}
}
