package common.network.layers.layers;

import common.network.layers.LayersMain;

public class NormLayer extends Layer{

	//https://www.pinecone.io/learn/batch-layer-normalization/
	//https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
	
	private float lastStdev;
	private float lastDeltas[][];
	private final int count;
	
	public NormLayer(Layer last) 
	{
		super(last, last.outputs);
		
		lastActivation = new float[outputs][depth];
		lastDeltas = new float[outputs][depth];
		count = outputs*depth;
	}

	@Override
	public float[][] activation(float[][] activations)
	{
		activations = lastLayer.getLastActivation();
		
		float average = 0;
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				if(Float.isNaN(activations[i][j]))
				{
					LayersMain.print(activations);
					System.out.println(activations.length);
					throw new IllegalArgumentException();
				}
				average += activations[i][j];
			}
		}
		average /= count;
		if(Float.isNaN(average))
		{
			throw new IllegalArgumentException();
		}
		
		float stdevSquared = 0;
		
		for(int i = 0; i < outputs; i++)
		{
			if(masks[i][0])
			{
				continue;
			}
			for(int j = 0; j < depth; j++)
			{
				lastDeltas[i][j] = activations[i][j] - average;
				stdevSquared += lastDeltas[i][j]*lastDeltas[i][j];
			}
		}
		
		stdevSquared /= count;
		lastStdev = (float)Math.sqrt(stdevSquared);
		if(Float.isNaN(stdevSquared))
		{
			throw new IllegalArgumentException();
		}
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				lastActivation[i][j] = lastDeltas[i][j] / lastStdev;
			}
		}
		
		return lastActivation;
	}

	@Override//VERIFIED
	public void backprop() {
		//See Jacobian section at following link:
		//https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
		float[][] nextErrorWeighted = getGradient();
		clearGradients();
		
		if(lastStdev == 0)
		{
			throw new IllegalArgumentException();
		}
		float stdevCubed = lastStdev * lastStdev * lastStdev;
		float negativeOneOverNStdevCubed = -1/(count * stdevCubed);
		float nMinusOneOverNStdev = (count - 1) / (count * lastStdev);
		float negativeOneOverNStdev = -1 / (count * lastStdev);
		
		if(Float.isNaN(stdevCubed))
		{
			throw new IllegalArgumentException();
		}
		
		float[][] thisError = new float[outputs][depth];
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				float deltaTimesNegativeOneOverNStdevCubed = lastDeltas[i][j] * negativeOneOverNStdevCubed;
				if(Float.isNaN(nextErrorWeighted[i][j]))
				{
					throw new IllegalArgumentException();
				}
				for(int k = i; k < outputs; k++)
				{
					for(int l = j; l < depth; l++)
					{
						float derivativeOfijWithRespectTokl = deltaTimesNegativeOneOverNStdevCubed * lastDeltas[k][l] + ((i == k && j == l) ? nMinusOneOverNStdev : negativeOneOverNStdev);
						thisError[i][j] += derivativeOfijWithRespectTokl * nextErrorWeighted[k][l];
						if(!(i == k && j == l))//dij/dkl == dkl/dij
						{
							thisError[k][l] += derivativeOfijWithRespectTokl * nextErrorWeighted[i][j];
						}
					}
				}
			}
		}
		
		lastLayer.reportGradient(thisError);
	}


	@Override
	public String name() {
		return "Norm";
	}
}
