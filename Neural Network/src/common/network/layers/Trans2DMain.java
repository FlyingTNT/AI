package common.network.layers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.TransformerModel2D;

public class Trans2DMain {

	public static void main(String[] args) {
		TransformerModel2D model = new TransformerModel2D(0.005f, 8, 8, null, new int[]{7, 1}, 8, new int[]{8}, new int[]{8, 2}, 2, 6);
		
		SimpleMatrix input1 = new SimpleMatrix(new double[][]{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
		SimpleMatrix output1 = new SimpleMatrix(new double[][]{{8, 2}, {7, 1}, {6, 2}, {5, 1}, {4, 2}, {3, 1}, {2, 2}, {1, 1}});
		SimpleMatrix output2 = new SimpleMatrix(new double[][]{{1, 1}, {2, 2}, {3, 1}, {4, 2}, {5, 1}, {6, 2}, {7, 1}, {8, 2}});
		SimpleMatrix input2 = new SimpleMatrix(new double[][]{{7}, {6}, {5}, {4}, {3}, {2}, {1}, {0}});
		SimpleMatrix[][] dataset = new SimpleMatrix[][]{{input1, output1}, {input2, output2}};
		
		float cost;
		int count = 0;
		
		for(int i = 0; count < 5; i++)
		{
			cost = model.epoch(dataset);
			System.out.println("Epoch " + i + " Cost: " + cost);
			if(cost < 1)
				count++;
			else
				count = 0;
		}
		
		model.beamSearch(input1, 1).print();
		model.beamSearch(input2, 1).print();
		
		File out = new File("C:\\AIClub\\Test\\model2d.txt");
		try {
			FileWriter writer = new FileWriter(out);
			writer.write(model.stringify());
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String string = null;
		try {
			string = Files.readString(Paths.get("C:\\AIClub\\Test\\model2d.txt"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		TransformerModel2D model2 = TransformerModel2D.load(string);
		
		model2.beamSearch(input1, 10).print();
		model2.beamSearch(input2, 10).print();
	}
}
