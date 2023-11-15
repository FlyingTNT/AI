package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

public class NormLayer extends Layer{

	//https://www.pinecone.io/learn/batch-layer-normalization/
	//https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
	
	private double lastStdev;
	private SimpleMatrix lastDeltas;
	private final int count;
	
	public NormLayer(Layer last) 
	{
		super(last, last.outputs);
		
		lastActivation = new SimpleMatrix(new float[outputs][depth]);
		lastDeltas = new SimpleMatrix(new float[outputs][depth]);
		count = outputs*depth;
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix activations)
	{
		masks = lastLayer.getMasks();
		activations = lastLayer.getLastActivation();
		
		double average = activations.elementSum();
		
		average /= count;
		
		if(Double.isNaN(average))
		{
			throw new IllegalArgumentException();
		}
		
		lastDeltas = activations.minus(average);
		
		lastStdev = lastDeltas.normF() / Math.sqrt(count);
		if(Double.isNaN(lastStdev))
		{
			throw new IllegalArgumentException();
		}
		
		lastActivation = lastDeltas.divide(lastStdev);
		
		return lastActivation;
	}

	@Override//VERIFIED
	public void backprop() {
		//See Jacobian section at following link:
		//https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
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
		//*/
		
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
