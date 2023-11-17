package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * A layer that takes the norm over the entire input. That is, modifies the input so that
 * its mean is 0 and its variance is 1.
 * @author C. Cooper
 */
public class NormLayer extends Layer{

	//https://www.pinecone.io/learn/batch-layer-normalization/
	//https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
	
	private double lastStdev;//The stdev of the last activation
	private SimpleMatrix lastDeltas;//A matrix of (input - average) from the inputs of the last activation
	private final int count;//The total number of inputs to this layer = inputs * depth.
	
	public NormLayer(Layer last) 
	{
		super(last, last.outputs);
		
		lastActivation = new SimpleMatrix(new float[outputs][depth]);
		lastDeltas = new SimpleMatrix(new float[outputs][depth]);
		count = outputs*depth;
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix activations, boolean isInference)
	{
		masks = lastLayer.getMasks();//This layer doesn't do masking, but pulls the masks forward in case the next layer does.
		activations = lastLayer.getLastActivation();//The last layer's activation
		
		double average = activations.elementSum();
		average /= count;
		
		if(Double.isNaN(average))//This gets hit whenever *something* is wrong with the model.
		{
			throw new IllegalArgumentException();
		}
		
		lastDeltas = activations.minus(average);//Subtracts the average from each element in activations and stores it in lastDeltas
		
		//Stdev = sqrt(sum((input - average for each input)^2)  / numberOfInputs)
		//      = sqrt(sum((input - average for each input)^2)) / sqrt(numberOfInputs)
		//      = lastDeltas.normF()                            / sqrt(count)
		lastStdev = lastDeltas.normF() / Math.sqrt(count);//Normf is the 'length' of the matrix. 
		if(Double.isNaN(lastStdev))
		{
			throw new IllegalArgumentException();
		}
		
		lastActivation = lastDeltas.divide(lastStdev);//Divides each item in lastDeltas by the stdev (produces matrix w/ mean 0 and variance 1)
		
		return lastActivation;
	}

	@Override//VERIFIED
	public void backprop() {
		//See Jacobian section at following link:
		//https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
		//Just look at the link I'm not explaining this
		SimpleMatrix nextErrorWeighted = getGradient();
		clearGradients();
		
		if(lastStdev == 0)
		{
			throw new IllegalArgumentException();
		}
		double stdevCubed = lastStdev * lastStdev * lastStdev;
		double negativeOneOverNStdevCubed = -1/(count * stdevCubed);
		double nMinusOneOverNStdev = (count - 1) / (count * lastStdev);
		double negativeOneOverNStdev = -1 / (count * lastStdev);
		double oneOverStdev = 1/lastStdev;
		
		//*
		SimpleMatrix base = lastDeltas.scale(negativeOneOverNStdevCubed);
		base = base.elementMult(nextErrorWeighted);
		
		double sum = base.elementSum();
		double sum2 = nextErrorWeighted.scale(negativeOneOverNStdev).elementSum();
		
		SimpleMatrix mod = nextErrorWeighted.scale(oneOverStdev);
		
		SimpleMatrix out = lastDeltas.scale(sum);
		out = out.plus(sum2);
		out = out.plus(mod);
		
		lastLayer.reportGradient(out);
		
		//*/This is a naive calculation of the gradient that takes O(n^4) (the one above takes O(n^2))
		
		/*
		if(Double.isNaN(stdevCubed))
		{
			throw new IllegalArgumentException();
		}
		
		float[][] thisError = new float[outputs][depth];
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				double deltaTimesNegativeOneOverNStdevCubed = lastDeltas.get(i, j) * negativeOneOverNStdevCubed;
				if(Double.isNaN(nextErrorWeighted.get(i, j)))
				{
					throw new IllegalArgumentException();
				}
				for(int k = i; k < outputs; k++)
				{
					for(int l = j; l < depth; l++)
					{
						double derivativeOfijWithRespectTokl = deltaTimesNegativeOneOverNStdevCubed * lastDeltas.get(k, l) + ((i == k && j == l) ? nMinusOneOverNStdev : negativeOneOverNStdev);
						thisError[i][j] += derivativeOfijWithRespectTokl * nextErrorWeighted.get(k, l);
						if(!(i == k && j == l))//dij/dkl == dkl/dij
						{
							thisError[k][l] += derivativeOfijWithRespectTokl * nextErrorWeighted.get(i, j);
						}
					}
				}
			}
		}
		
		lastLayer.reportGradient(new SimpleMatrix(thisError));
		*/
	}


	@Override
	public String name() {
		return "Norm";
	}
	
	@Override
	public String stringify() {
		return getId() + " " + lastLayer.getId() + " " + inputs + " " + depth + " " + outputs;
	}
	
	/**
	 * Creates a NormLayer from a String produced by this class's {@link #stringify()} method.
	 * <br><br>
	 * Only the first two numbers (thisId and lastLayerId) in the string are actually used.
	 * @param string A string produced by {@link #stringify()}
	 * @param model The model this layer belongs to.
	 * @param pos The position of this layer in the model (not used)
	 * @return A NormLayer based on the given string.
	 */
	public static NormLayer load(String string, LayersModel model, int pos) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastId = scanner.nextInt();
		scanner.close();
		Layer lastLayer = model.getLayerByID(lastId);
		NormLayer out = new NormLayer(lastLayer);
		out.setId(id);
		return out;
	}
}
