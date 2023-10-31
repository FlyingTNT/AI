package common.network.layers.layers;

import common.network.layers.LayersMain;

public class PositionalEncoding extends Layer{

	float[][] matrix;
	
	public PositionalEncoding(EmbeddingLayer last) {
		super(last.outputs, last.outputs);
		lastLayer = last;
		depth = last.depth;
		
		matrix = new float[outputs][depth];
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				matrix[i][j] = positionalEmbedding(i, depth, j);
			}
		}
		
		System.out.println("Positional:");
		LayersMain.print(matrix);
		lastActivation = new float[outputs][depth];
	}

	@Override
	public float[][] activation(float[][] input) {
		input = lastLayer.getLastActivation();
		masks = lastLayer.getMasks();
		
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				lastActivation[i][j] = input[i][j] + matrix[i][j];
			}
		}
		
		return lastActivation;
	}

	@Override
	public void backprop() {
		lastLayer.reportGradient(getGradient());
		clearGradients();
	}

	@Override
	public String name() {
		return "Positional Encoding";
	}

	public static float[][] generatePositionalEncoding(int sequenceLength, int embeddingDepth) {
        float[][] positionalEncoding = new float[sequenceLength][embeddingDepth];

        for (int pos = 0; pos < sequenceLength; pos++) {
            for (int i = 0; i < embeddingDepth; i++) {
                double angle = pos / Math.pow(10000, 2.0 * i / embeddingDepth);
                if (i % 2 == 0) {
                    positionalEncoding[pos][i] = (float) Math.sin(angle);
                } else {
                    positionalEncoding[pos][i] = (float) Math.cos(angle);
                }
            }
        }

        return positionalEncoding;
    }
	
	static float positionalEmbedding(int position, int embeddingDepth, int embeddingDepthPosition)
	{
		double inner = position/  Math.pow(10000, (embeddingDepthPosition/2) / (double)embeddingDepth);
		
		if(embeddingDepthPosition % 2 == 0)
		{
			return (float)Math.sin(inner);
		}else {
			return (float)Math.cos(inner);
		}
	}
}
