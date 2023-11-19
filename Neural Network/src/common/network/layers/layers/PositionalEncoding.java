package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

import common.network.layers.models.LayersModel;

/**
 * Positional Encoding layer.
 * @author C. Cooper
 */
public class PositionalEncoding extends Layer{

	private final SimpleMatrix matrix;//The positional encoding matrix
	
	/**
	 * Creates a PositionalEncoding layer that applies the encoding to the given {@link EmbeddingLayer}
	 * @param last The {@link EmbeddingLayer} to apply positional encoding to.
	 */
	public PositionalEncoding(EmbeddingLayer last) {
		super(last.outputs, last.outputs);
		lastLayer = last;
		depth = last.depth;
		matrix = new SimpleMatrix(generatePositionalEncoding(outputs, depth));//Pre-generates the encoding matrix
		setGradientSize(inputs, depth);
	}

	@Override
<<<<<<< HEAD
	public float[][] activation(float[][] input) {
		input = lastLayer.getLastActivation();
		
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				lastActivation[i][j] = input[i][j] + matrix[i][j];
			}
		}
		
=======
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		masks = lastLayer.getMasks();//Pulls the masks forward
		lastActivation = lastLayer.getLastActivation().plus(matrix);//Adds the encoding matrix to the last layer's activation	
>>>>>>> refs/remotes/origin/ejml
		return lastActivation;
	}

	@Override
	public void backprop() {
		lastLayer.reportGradient(getGradient());//Just sends the gradient backwards (this layer is constant so it doesn't affect the gradient)
		clearGradients();
	}

	@Override
	public String name() {
		return "Positional Encoding";
	}
	
	@Override
	public String className() {
		return "PositionalEncoding";
	}
	
	//I'm not documenting the generation functions because I'm not convinced that any of them work. Imo the best is 2, but Breck told me to use the one I'm currently using.
	
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
	
	@Override
	public String stringify() {
		return getId() + " " + lastLayer.getId() + " " + inputs + " " + depth;
	}
	
	/**
	 * Loads a PositionalEncoding layer based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static PositionalEncoding load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		scanner.close();
		
		PositionalEncoding out = new PositionalEncoding((EmbeddingLayer)model.getLayerByID(lastID));
		out.setId(id);
		
		return out;
	}
}
