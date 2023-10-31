package common.network.layers;

import common.network.layers.models.TransformerModel;

public class TransMain3 {
	static int SEQUENCE_LENGTH = 530;
	static int EMBED_DEPTH = 12;
	static int HEADS = 4;
	static int TRANSFORMER_STACK_SIZE = 6;
	static int VOCAB_SIZE = 183;
	static float LEARNING_RATE = 0.005f;
	
	static String DATA_FOLDER = "C:\\AIClub\\Code\\Small Dataset\\Tokenized";
	
	public static void main(String[] args) {
		TransformerModel transformer = new TransformerModel(LEARNING_RATE, SEQUENCE_LENGTH, EMBED_DEPTH, VOCAB_SIZE, HEADS, TRANSFORMER_STACK_SIZE);
		
		float[][] problemsTokenized = DatasetLoader.loadProblems(DATA_FOLDER);
		float[][] submissionsTokenized = DatasetLoader.loadSubmissions(DATA_FOLDER);
		
		//LayersMain.print(problemsTokenized);
		//LayersMain.print(submissionsTokenized);
		
		//System.out.println(problemsTokenized[0].length);
		//System.out.println(submissionsTokenized[0].length);
		
		//float max = 0;
		
		float[][][][] dataset = new float[problemsTokenized.length][2][SEQUENCE_LENGTH][1];
		
		for(int i = 0; i < problemsTokenized.length; i++)
		{
			for(int j = 0; j < SEQUENCE_LENGTH; j++)
			{
				//if(problemsTokenized[i][j] > max)
				//	max = problemsTokenized[i][j];
				//if(submissionsTokenized[i][j] > max)
				//	max = submissionsTokenized[i][j];
				dataset[i][0][j][0] = problemsTokenized[i][j];
				dataset[i][1][j][0] = submissionsTokenized[i][j];
			}
		}
		
		//System.out.println(max);
		for(int i = 0; i < 10; i++)
			System.out.println("Epoch: " + i + ", Cost: " + transformer.epoch(dataset));
	}
}
