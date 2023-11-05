package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

public class PositionalEncoding extends Layer{

	SimpleMatrix matrix;
	
	public PositionalEncoding(EmbeddingLayer last) {
		super(last.outputs, last.outputs);
		lastLayer = last;
		depth = last.depth;
		matrix = new SimpleMatrix(generatePositionalEncoding2(outputs, depth));
		setGradientSize(inputs, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		masks = lastLayer.getMasks();
		lastActivation = lastLayer.getLastActivation().plus(matrix);	
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
	
	public static double[][] generatePositionalEncoding2(int sequenceLength, int embeddingDepth) {
        double[][] positionalEncoding = new double[sequenceLength][embeddingDepth];

        for (int pos = 0; pos < sequenceLength; pos++) {
            for (int i = 0; i < embeddingDepth; i+=2) {
                double angle = Math.exp(i * -Math.log(10000d) / embeddingDepth);
                if(Double.isNaN(angle))
                	throw new IllegalAccessError();
                positionalEncoding[pos][i] = Math.sin(pos*angle);
                positionalEncoding[pos][i+1] = Math.cos(pos*angle);
            }
        }

        return positionalEncoding;
    }
	
	public static double[][] generatePositionalEncoding(int sequenceLength, int embeddingDepth) {
	    double[][] positionalEncoding = new double[sequenceLength][embeddingDepth];

	    for (int pos = 0; pos < sequenceLength; pos++) {
	        for (int i = 0; i < embeddingDepth; i++) {
	            double angle = pos / Math.pow(10000, (2 * i) / (double) embeddingDepth);
	            positionalEncoding[pos][i] = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);
	        }
	    }

	    return positionalEncoding;
	}

	
	static SimpleMatrix positionalEmbedding(int outputs, int depth)
	{
		return new SimpleMatrix(new double[outputs][depth]).elementOp(new ElementOpReal() {
			
			@Override
			public double op(int row, int col, double value) {
				return positionalEmbedding(row, depth, col);
			}
		});
	}
	
	static double positionalEmbedding(int position, int embeddingDepth, int embeddingDepthPosition)
	{
		double inner = position/  Math.pow(10000, (embeddingDepthPosition/2) / (double)embeddingDepth);
		
		if(embeddingDepthPosition % 2 == 0)
		{
			return Math.sin(inner);
		}else {
			return Math.cos(inner);
		}
	}
}
