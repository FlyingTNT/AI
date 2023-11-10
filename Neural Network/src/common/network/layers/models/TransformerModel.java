package common.network.layers.models;

import java.util.ArrayList;
import org.ejml.simple.SimpleMatrix;
import common.network.layers.Activation;
import common.network.layers.Cost;
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
		
		encoders[0] = new Encoder(encoderIn, heads, true);
		decoders[0] = new Decoder(decoderIn, encoders[0], heads, true, true);
		
		for(int i = 1; i < layers; i++)
		{
			encoders[i] = new Encoder(encoders[i-1], heads, true);
			decoders[i] = new Decoder(decoders[i-1], encoders[i], heads, false, true);
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
	public float epoch(SimpleMatrix[]... trainingSet) 
	{//Training set: [item number][input/target][i/o position axis 0][i/o position axis 1]
		decoders[0].setInference(false);
		float costSum = 0;
		for(int i = 0; i < trainingSet.length; i++)
		{
			encoderIn.activation(trainingSet[i][0]);
			decoderIn.activation(shift(trainingSet[i][1], BOS));
			//System.out.println("Decoder In:");
			//shift(trainingSet[i][1], BOS).print();
			
			for(int j = 0; j < encoders.length; j++)
			{
				//System.out.println("Layer " + j);
				encoders[j].activation(null);
				decoders[j].activation(null);
			}
			
			rotate.activation(null);
			outputLinear.activation(null);
			output.activation(null);
			
			SimpleMatrix modelOut = output.getLastActivation();
			
			//System.out.println("Decoder Out:");
			//modelOut.print();
			
			costSum += cost.cost(modelOut, trainingSet[i][1]);
			
			SimpleMatrix lastErrorIn = cost.derivative(modelOut, trainingSet[i][1]);
			
			//lastErrorIn.print();
			
			output.reportGradient(lastErrorIn);
			
			//System.out.println("===========BACK============ ");
			
			for(int j = model.length - 1; j >= 0; j--)
			{
				model[j].backprop();
			}
			//int s = 0/0;
		}
		
		return costSum / trainingSet.length;
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix input) {
		encoderIn.activation(input);
		decoders[0].setInference(true);
		
		//System.out.println("Inferencing:");
		//input.print();
		
		for(int i = 0; i < encoders.length; i++)
		{
			encoders[i].activation(null);
		}
		
		SimpleMatrix currentIn = SimpleMatrix.filled(sequenceLength+1, 1, PADDING);
		currentIn.set(0, BOS);
		
		for(int i = 0; i < sequenceLength; i++)
		{
			//System.out.println("\nDecoder input:");
			//currentIn.print();
			decoderIn.activation(currentIn);
			for(int j = 0; j < decoders.length; j++)
			{
				decoders[j].activation(null);
			}
			
			rotate.activation(null);
			outputLinear.activation(null);
			output.activation(null);
			
			//System.out.println("\nDecoder Output:");
			//output.getLastActivation().print();
			
			int newToken = NetworkMath.argmax(output.getLastActivation().getRow(i));
			currentIn.set(i+1, newToken);
		}
		return currentIn;
	}
	
	public void test(SimpleMatrix[]... trainingSet)
	{
		System.out.println("===============TESTING===============");
		decoders[0].setInference(false);
		float costSum = 0;
		for(int i = 0; i < trainingSet.length; i++)
		{
			encoderIn.activation(trainingSet[i][0]);//ENCODER IN
			decoderIn.activation(shift(trainingSet[i][1], BOS));//DECODER IN
			
			//SimpleMatrix in = SimpleMatrix.filled(trainingSet[i][1].getNumRows(), 1, -1);
			//in.set(0, BOS);
			
			//decoderIn.activation(in);
			
			System.out.println("\nEncoder in:");
			trainingSet[i][0].print();
			
			System.out.println("\nDecoder in:");
			//in.print();
			shift(trainingSet[i][1], BOS).print();
			
			for(int j = 0; j < encoders.length; j++)
			{
				encoders[j].activation(null);
				decoders[j].activation(null);
			}
			
			rotate.activation(null);
			outputLinear.activation(null);
			output.activation(null);
			
			SimpleMatrix modelOut = output.getLastActivation();
			
			System.out.println("\nModel Out:");
			modelOut.print();
			
			System.out.print("Cost: ");
			
			double costVal = cost.cost(modelOut, trainingSet[i][1]);
			
			System.out.println(costVal);
			
			costSum += costVal;
		}
		
		System.out.println("\nTotal Cost: " + costSum / trainingSet.length);
	}
	
	public SimpleMatrix beamSearch(SimpleMatrix input, int width)
	{
		encoderIn.activation(input);
		decoders[0].setInference(false);
		
		for(int i = 0; i < encoders.length; i++)
		{
			encoders[i].activation(null);
		}
		
		SimpleMatrix currentIn = SimpleMatrix.filled(sequenceLength+1, 1, PADDING);
		currentIn.set(0, BOS);
		
		SimpleMatrix[] currentBest = new SimpleMatrix[width];
		Double[] currentProbabilities = new Double[width];
		SimpleMatrix out = doDecoder(currentIn);
		
		ArrayList<SimpleMatrix> best = new ArrayList<>();
		ArrayList<Double> topProbabilities = new ArrayList<>();
		
		for(int i = 0; i < width; i++)
		{
			best.add(null);
			topProbabilities.add(Double.NEGATIVE_INFINITY);
		}
		
		for(int i = 0; i < vocabSize; i++)
		{
			for(int j = 0; j < width; j++)
			{
				if(Math.log(out.get(0, i)) > topProbabilities.get(j) || topProbabilities.get(j) == Double.NEGATIVE_INFINITY)
				{
					currentIn = SimpleMatrix.filled(sequenceLength+1, 1, PADDING);
					currentIn.set(0, BOS);
					currentIn.set(1, i);
					
					best.add(j, currentIn);
					topProbabilities.add(j, Math.log(out.get(0, i)));
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
				if(currentBest[j] == null)
					continue;
				out = doDecoder(currentBest[j]);
				
				for(int k = 0; k < vocabSize; k++)
				{
					double probability = Math.log(out.get(i-1, k)) + currentProbabilities[j];
					
					for(int l = 0; l < width; l++)
					{
						if(probability > topProbabilities.get(l) || topProbabilities.get(l) == Double.NEGATIVE_INFINITY)
						{
							topProbabilities.add(l, probability);
							SimpleMatrix newArray = currentBest[j].copy();
							newArray.set(i, 0, k);
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
	
	private SimpleMatrix doDecoder(SimpleMatrix input)
	{
		decoderIn.activation(input);
		for(int i = 0; i < decoders.length; i++)
			decoders[i].activation(null);
		rotate.activation(null);
		outputLinear.activation(null);
		output.activation(null);
		return output.getLastActivation();
	}
	
	public static SimpleMatrix shift(SimpleMatrix input, int bosToken)
	{
		SimpleMatrix input1 = input.copy();
		for(int i = input1.getNumRows() - 1; i >= 1; i--)
			input1.set(i, input1.get(i-1));
		input1.set(0, bosToken);
		return input1;
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
}
