package common.network.layers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.TransformerModel2D;

public class TransMain4 {
	static int TOKEN_EMBED_DEPTH = 12;
	static int TYPE_EMBED_DEPTH = 12;
	static int EMBED_DEPTH = TOKEN_EMBED_DEPTH + TYPE_EMBED_DEPTH;
	static int HEADS = 4;
	static int TRANSFORMER_STACK_SIZE = 6;
	static int ENCODER_VOCAB_SIZE = 1220;
	static int ENCODER_SEQUENCE_LENGTH = 1378;
	static int DECODER_VOCAB_SIZE = 339;
	static int DECODER_TYPE_SIZE = 61;
	static int DECODER_SEQUENCE_LENGTH = 716;
	static float LEARNING_RATE = 0.001f;
	
	static String DATA_FOLDER =  "C:\\AIClub\\Code\\Large Dataset\\Tokenized2";
	
	public static void main(String[] args) {		
		float[][] problemsTokenized = DatasetLoader.loadProblems(DATA_FOLDER);
		float[][][] submissionsTokenized = DatasetLoader.loadSubmissions2d(DATA_FOLDER);
		
		//LayersMain.print(problemsTokenized);
		//LayersMain.print(submissionsTokenized);
		
		//System.out.println(problemsTokenized[0].length);
		//System.out.println(submissionsTokenized[0].length);
		
		//float max = 0;
		
		SimpleMatrix[][] dataset = new SimpleMatrix[problemsTokenized.length][2];
		
		for(int i = 0; i < problemsTokenized.length; i++)
		{
			dataset[i][0] = new SimpleMatrix(ENCODER_SEQUENCE_LENGTH, 1);
			dataset[i][1] = new SimpleMatrix(DECODER_SEQUENCE_LENGTH, submissionsTokenized[0].length);
			for(int j = 0; j < ENCODER_SEQUENCE_LENGTH; j++)
			{
				dataset[i][0].set(j, 0, problemsTokenized[i][j]);
			}
			for(int j = 0; j < DECODER_SEQUENCE_LENGTH; j++)
			{
				for(int k = 0; k < submissionsTokenized[0].length; k++)
				{
					dataset[i][1].set(j, k, submissionsTokenized[i][k][j]);
				}
			}
		}
		
		System.out.println("Loaded");
		TransformerModel2D model = new TransformerModel2D(LEARNING_RATE, ENCODER_SEQUENCE_LENGTH, DECODER_SEQUENCE_LENGTH, null, new int[] {TOKEN_EMBED_DEPTH, TYPE_EMBED_DEPTH}, EMBED_DEPTH, new int[] {ENCODER_VOCAB_SIZE}, new int[] {DECODER_VOCAB_SIZE, DECODER_TYPE_SIZE}, HEADS, 6);
		int count = 0;
		for(int i = 0; count < 5; i++)
		{
			float cost = model.epoch(dataset);
			System.out.println("Epoch " + i + ", Cost: " + cost);
			File out = new File("C:\\AIClub\\Models\\transformer_epoch_" + i + ".txt");
			try {
				FileWriter writer = new FileWriter(out);
				writer.write(model.stringify());
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}	
		}
	}

}
