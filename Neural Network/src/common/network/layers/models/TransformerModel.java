package common.network.layers.models;

import java.util.ArrayList;

import common.network.layers.Activation;
import common.network.layers.Cost;
import common.network.layers.LayersMain;
import common.network.layers.layers.Decoder;
import common.network.layers.layers.Encoder;
import common.network.layers.layers.Layer;
import common.network.layers.layers.RotationLayer;
import common.network.layers.layers.StandardLayer;
import common.network.layers.layers.TransformerInput;
import common.network.math.NetworkMath;

public class TransformerModel extends LayersNetwork{

	int sequenceLength;
	int embeddingDepth;
	int vocabSize;
	int heads;
	int EOS;
	int BOS;
	int PADDING;
	
	TransformerInput encoderIn;
	TransformerInput decoderIn;
	Encoder[] encoders;
	Decoder[] decoders;
	
	RotationLayer rotate;
	StandardLayer outputLinear;
	RotationLayer output;
	
	public TransformerModel(float learningRate, int sequenceLength, int embeddingDepth, int vocabSize, int heads, int layers) {
		this.sequenceLength = sequenceLength;
		this.embeddingDepth = embeddingDepth;
		this.vocabSize = vocabSize;
		this.EOS = this.vocabSize - 1;
		this.heads = heads;
		this.learningRate = learningRate;
		this.cost = Cost.SPARSE_CATEGORICAL_CROSS_ENTROPY;
		
		PADDING = -1;
		BOS = 0;
		EOS = vocabSize+1;
		this.vocabSize += 2;
		vocabSize = this.vocabSize;
		
		encoderIn = new TransformerInput(sequenceLength, embeddingDepth, vocabSize);
		decoderIn = new TransformerInput(sequenceLength, embeddingDepth, vocabSize);
		
		encoders = new Encoder[layers];
		decoders = new Decoder[layers];
		
		encoders[0] = new Encoder(encoderIn, heads);
		decoders[0] = new Decoder(decoderIn, encoders[0], heads, true);
		
		for(int i = 1; i < layers; i++)
		{
			encoders[i] = new Encoder(encoders[i-1], heads);
			decoders[i] = new Decoder(decoders[i-1], encoders[i], heads, false);
		}
		
		rotate = new RotationLayer(decoders[decoders.length - 1]);
		outputLinear = new StandardLayer(rotate, vocabSize, Activation.SOFTMAX);
		output = new RotationLayer(outputLinear);
		
		model = new Layer[2*(layers+1) + 3];
		model[0] = encoderIn;
		model[1] = decoderIn;
		
		for(int i = 1; i <= layers; i++)
		{
			model[2*i] = encoders[i-1];
			model[(2*i)+1] = decoders[i-1];
		}
		
		model[model.length - 1] = output;
		model[model.length - 2] = outputLinear;
		model[model.length - 3] = rotate;
		
		for(int i = 0; i < model.length; i++)
		{
			model[i].setModel(this);
		}
	}
	
	@Override
	public float epoch(float[][][]... trainingSet) 
	{//Training set: [item number][input/target][i/o position axis 0][i/o position axis 1]
		decoders[0].setMasking(true);
		float costSum = 0;
		for(int i = 0; i < trainingSet.length; i++)
		{
			encoderIn.activation(trainingSet[i][0]);
			decoderIn.activation(shift(trainingSet[i][1], BOS));
			
			for(int j = 0; j < encoders.length; j++)
			{
				//System.out.println("Layer " + j);
				encoders[j].activation(null);
				decoders[j].activation(null);
			}
			
			rotate.activation(null);
			outputLinear.activation(null);
			output.activation(null);
			
			float[][] modelOut = output.getLastActivation();
			
			costSum += cost.cost(modelOut, trainingSet[i][1]);
			
			float[][] lastErrorIn = cost.derivative(modelOut, trainingSet[i][1]);
			
			output.reportGradient(lastErrorIn);
			
			for(int j = model.length - 1; j >= 0; j--)
			{
				model[j].backprop();
			}
		}
		
		return costSum / trainingSet.length;
	}
	
	@Override
	public float[][] feedForward(float[][] input) {
		encoderIn.activation(input);
		decoders[0].setMasking(false);
		decoderIn.setMasking(false);
		
		System.out.println("Inferencing:");
		LayersMain.print(input);
		
		for(int i = 0; i < encoders.length; i++)
		{
			encoders[i].activation(null);
		}
		
		float[][] currentIn = new float[sequenceLength+1][1];
		currentIn[0][0] = BOS;
		///*
		for(int i = 1; i < sequenceLength+1; i++)
		{
			currentIn[i][0] = -1;
		}//*/
		
		for(int i = 0; i < sequenceLength; i++)
		{
			System.out.println("\nDecoder input:");
			LayersMain.print(currentIn);
			decoderIn.activation(currentIn);
			for(int j = 0; j < decoders.length; j++)
			{
				decoders[j].activation(null);
			}
			
			rotate.activation(null);
			outputLinear.activation(null);
			output.activation(null);
			
			System.out.println("\nDecoder Output:");
			LayersMain.print(output.getLastActivation());
			
			int newToken = NetworkMath.argmax(output.getLastActivation()[i]);
			currentIn[i+1][0] = newToken;
		}
		return currentIn;
	}
	
	public void test(float[][][]... trainingSet)
	{
		System.out.println("===============TESTING===============");
		decoders[0].setMasking(false);
		float costSum = 0;
		for(int i = 0; i < trainingSet.length; i++)
		{
			encoderIn.activation(trainingSet[i][0]);//ENCODER IN
			decoderIn.activation(shift(trainingSet[i][1], BOS));//DECODER IN
			
			System.out.println("\nEncoder in:");
			LayersMain.print(trainingSet[i][0]);
			
			System.out.println("\nDecoder in:");
			LayersMain.print(trainingSet[i][1]);
			
			for(int j = 0; j < encoders.length; j++)
			{
				encoders[j].activation(null);
				decoders[j].activation(null);
			}
			
			rotate.activation(null);
			outputLinear.activation(null);
			output.activation(null);
			
			float[][] modelOut = output.getLastActivation();
			
			System.out.println("\nModel Out:");
			LayersMain.print(modelOut);
			
			System.out.print("Cost: ");
			
			System.out.println(cost.cost(modelOut, trainingSet[i][1]));
			
			costSum += cost.cost(modelOut, trainingSet[i][1]);
		}
		
		System.out.println("\nTotal Cost: " + costSum / trainingSet.length);
	}
	
	public float[][] beamSearch(float[][] input, int width)
	{
		encoderIn.activation(input);
		decoders[0].setMasking(false);
		decoderIn.setMasking(false);
		
		for(int i = 0; i < encoders.length; i++)
		{
			encoders[i].activation(null);
		}
		
		float[][] currentIn = new float[sequenceLength+1][1];
		currentIn[0][0] = BOS;
		///*
		for(int i = 1; i < sequenceLength+1; i++)
		{
			currentIn[i][0] = -1;
		}//*/
		
		float[][][] currentBest = new float[width][sequenceLength + 1][1];
		Double[] currentProbabilities = new Double[width];
		float[][] out = doDecoder(currentIn);
		
		ArrayList<float[][]> best = new ArrayList<>();
		ArrayList<Double> topProbabilities = new ArrayList<>();
		
		best.add(null);
		topProbabilities.add(Double.NEGATIVE_INFINITY);
		
		for(int i = 0; i < vocabSize; i++)
		{
			for(int j = 0; j < width; j++)
			{
				if(Math.log(out[0][i]) > topProbabilities.get(j) || topProbabilities.get(j) == Double.NEGATIVE_INFINITY)
				{
					currentIn = new float[sequenceLength+1][1];
					currentIn[0][0] = BOS;
					currentIn[1][0] = i;
					///*
					for(int k = 2; k < sequenceLength+1; k++)
					{
						currentIn[k][0] = -1;
					}//*/
					best.add(j, currentIn);
					topProbabilities.add(j, Math.log(out[0][i]));
					if(topProbabilities.size() > width)
					{
						best.remove(topProbabilities.size() - 1);
						topProbabilities.remove(topProbabilities.size() - 1);
					}
					break;
				}
			}
		}
		
		currentBest = best.toArray(currentBest);
		currentProbabilities = topProbabilities.toArray(currentProbabilities);
		
		for(int i = 2; i < sequenceLength + 1; i++)
		{
			best.clear();
			topProbabilities.clear();
			best.add(null);
			topProbabilities.add(Double.NEGATIVE_INFINITY);
			
			for(int j = 0; j < width; j++)
			{
				out = doDecoder(currentBest[j]);
				
				for(int k = 0; k < vocabSize; k++)
				{
					double probability = Math.log(out[i-1][k]) + currentProbabilities[j];
					
					for(int l = 0; l < width; l++)
					{
						if(probability > topProbabilities.get(l) || topProbabilities.get(l) == Double.NEGATIVE_INFINITY)
						{
							topProbabilities.add(l, probability);
							float[][] newArray = new float[sequenceLength+1][1];
							for(int m = 0; m < i; m++)
								newArray[m][0] = currentBest[j][m][0];
							newArray[i][0] = k;
							best.add(l, newArray);
							if(topProbabilities.size() > width)
							{
								best.remove(topProbabilities.size() - 1);
								topProbabilities.remove(topProbabilities.size() - 1);
							}
							break;
						}
					}
				}
			}
			
			currentBest = best.toArray(currentBest);
			currentProbabilities = topProbabilities.toArray(currentProbabilities);
			/*
			System.out.println(i);
			for(int j = 0; j < width; j++) {
				System.out.println(currentProbabilities[j]);
				LayersMain.print(currentBest[j]);}
				*/
		}
		
		return currentBest[0];
	}
	
	private float[][] doDecoder(float[][] input)
	{
		decoderIn.activation(input);
		for(int i = 0; i < decoders.length; i++)
			decoders[i].activation(null);
		rotate.activation(null);
		outputLinear.activation(null);
		output.activation(null);
		return output.getLastActivation();
	}
	
	public static float[][] shift(float[][] input, int bosToken)
	{
		float[][] out = new float[input.length][1];
		out[0][0] = bosToken;
		for(int i = 1; i < input.length; i++)
		{
			out[i][0] = input[i-1][0];
		}
		return out;
	}
	
	@Override
	public String toString() {
		String out = "Model: (" + sequenceLength + ", " + 1 + ") -> (" + sequenceLength + ", " + vocabSize + ")\n";
		for(int i = 0; i < model.length; i++)
		{
			out += model[i] + "\n";
		}
		return out;
	}
	
	private static void toArray(ArrayList<float[][]> list, float[][][] array)
	{
		for(int i = 0; i < array.length; i++)
		{
			array[i] = list.get(i);
		}
	}
}
