package common.network.layers.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

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

public class TransformerModel2D extends LayersModel{
	int encoderSequenceLength;//The length of the token lists that may be given to the encoder.
	int decoderSequenceLength;//The length of the token lists that may be given to the decoder.
	int embeddingDepth;//The size of the embedding vectors
	int[] encoderVocabSizes;//The number of tokens in the encoder vocabulary
	int[] decoderVocabSizes;//The number of tokens in the decoder vocabulary
	int heads;//The number of heads in the attention layers
	int[] EOSES;//The End of Sequence token (generally the last token in the decoder vocab) - only in decoder
	int BOS;//The Beginning of Sequence token (generally 0) - only in decoder
	int PADDING;//The padding token (generally -1) - In both encoder and decoder
	
	TransformerInput encoderIn;//The encoder input layer
	TransformerInput decoderIn;//The decoder input layer
	Encoder[] encoders;//The encoder stack
	Decoder[] decoders;//The decoder stack
	
	/*
	 * The output of the decoder stack is sequence length x embed depth. The output of the model needs to be
	 * sequence length x vocab size. Standard linear layers can only expand their dimensions along the first axis,
	 * so to expand from (sequence length x embed depth) -> (sequence length x vocab size) we need to rotate
	 * (sequence length x embed depth) -> (embed depth x sequence length), then expand (embed depth x sequence length)
	 * -> (vocab size x sequence length), then roatate again (vocab size x sequence length) -> (sequence length x vocab size)
	 */
	RotationLayer rotate;//Rotates the decoder output
	StandardLayer[] outputLinears;//Expands (embed depth x sequence length)->(vocab size x sequence length) and takes the softmax
	RotationLayer[] outputs;//Rotates back to (sequence length x vocab size)
	
	/**
	 * Basic constructor for a transformer.
	 * @param learningRate The learning rate to use during training.
	 * @param encoderSequenceLength The sequence length for the encoder
	 * @param decoderSequenceLength The sequence length for the decoder
	 * @param embeddingDepth The size of the embedding vectors
	 * @param encoderVocabSize The number of tokens in the encoder vocab.
	 * @param decoderVocabSize The number of tokens in the decoder vocab. <i>Do not include BOS/EOS/padding tokens. The model accounts for them</i>
	 * @param heads The number of heads to use in the attention layers. Must be a factor of the embed depth
	 * @param layers The number of encoders/decoders to use.
	 */
	public TransformerModel2D(float learningRate, int encoderSequenceLength, int decoderSequenceLength, int[] encoderEmbeddingDepths, int[] decoderEmbeddingDepths, int totalEmbeddingDepth, int[] encoderVocabSizes, int[] decoderVocabSizes, int heads, int layers) {
		this.encoderSequenceLength = encoderSequenceLength;
		this.decoderSequenceLength = decoderSequenceLength;
		this.embeddingDepth = totalEmbeddingDepth;
		this.encoderVocabSizes = encoderVocabSizes;
		this.decoderVocabSizes = decoderVocabSizes;
		this.heads = heads;
		this.learningRate = learningRate;
		this.cost = Cost.SPARSE_CATEGORICAL_CROSS_ENTROPY;
		
		PADDING = -1;
		BOS = 0;
		EOSES = new int[decoderVocabSizes.length];
		
		for(int i = 0; i < decoderVocabSizes.length; i++)
			this.decoderVocabSizes[i] += 2;
		
		encoderIn = new TransformerInput(encoderSequenceLength, encoderEmbeddingDepths, totalEmbeddingDepth, encoderVocabSizes);
		decoderIn = new TransformerInput(decoderSequenceLength, decoderEmbeddingDepths, totalEmbeddingDepth, this.decoderVocabSizes);
		
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
		outputLinears = new StandardLayer[decoderVocabSizes.length];
		outputs = new RotationLayer[decoderVocabSizes.length];
		
		for(int i = 0; i < decoderVocabSizes.length; i++)
		{
			EOSES[i] = this.decoderVocabSizes[i] - 1;
			outputLinears[i] = new StandardLayer(rotate, this.decoderVocabSizes[i], Activation.SOFTMAX);
			outputs[i] = new RotationLayer(outputLinears[i]);
		}
		
		model = new Layer[2*(layers) + 3 + decoderVocabSizes.length * 2];//2 * num of encoders/decoders + encoder in/decoder in + rotate, linear, rotate
		model[0] = encoderIn;
		model[1] = decoderIn;
		
		for(int i = 1; i <= layers; i++)
		{
			model[2*i] = encoders[i-1];
			model[(2*i)+1] = decoders[i-1];
		}
		
		model[2 * layers + 2] = rotate;
		
		int current = 2 * layers + 3;
		for(int i = 0; i < decoderVocabSizes.length; i++)
		{
			model[current] = outputLinears[i];
			current++;
			model[current] = outputs[i];
			current++;
		}
		
		for(int i = 0; i < model.length; i++)
		{
			System.out.println(model[i]);
			model[i].setModel(this);
		}
	}
	
	/**
	 * Constructor used in the {@link #load(String)} function.
	 * @param encoderSequenceLength The sequence length for the encoder
	 * @param decoderSequenceLength The sequence length for the decoder
	 * @param embeddingDepth The size of the embedding vectors
	 * @param encoderVocabSize The number of tokens in the encoder vocab.
	 * @param decoderVocabSize The number of tokens in the decoder vocab. <i>DOES INCLUDE META TOKENS (not padding)</i>
	 * @param heads The number of heads to use in the attention layers. Must be a factor of the embed depth
	 * @param layers The number of encoders/decoders to use.
	 */
	private TransformerModel2D(int encoderSequenceLength, int decoderSequenceLength, int embeddingDepth, int[] encoderVocabSizes, int[] decoderVocabSizes, int heads, int layers)
	{
		this.encoderSequenceLength = encoderSequenceLength;
		this.decoderSequenceLength = decoderSequenceLength;
		this.embeddingDepth = embeddingDepth;
		this.encoderVocabSizes = encoderVocabSizes;
		this.decoderVocabSizes = decoderVocabSizes;
		EOSES = new int[decoderVocabSizes.length];
		for(int i = 0; i < decoderVocabSizes.length; i++)
			EOSES[i] = this.decoderVocabSizes[i] - 1;
		this.heads = heads;
		this.learningRate = 0;
		this.cost = Cost.SPARSE_CATEGORICAL_CROSS_ENTROPY;
		
		PADDING = -1;
		BOS = 0;
		
		encoders = new Encoder[layers];
		decoders = new Decoder[layers];
		
		outputLinears = new StandardLayer[decoderVocabSizes.length];
		outputs = new RotationLayer[decoderVocabSizes.length];
		
		model = new Layer[2*(layers) + 3 + decoderVocabSizes.length * 2];
	}

	
	@Override
	public float epoch(SimpleMatrix[]... trainingSet) 
	{//Training set: [item number][input/target][i/o position axis 0][i/o position axis 1]
		float costSum = 0;
		for(int i = 0; i < trainingSet.length; i++)//For each item in the training set,
		{
			encoderIn.activation(trainingSet[i][0], false);//Activates the encoder input on the training set's input
			decoderIn.activation(shift(trainingSet[i][1], BOS), false);//Activates the decoder input on the training set's target (shifted right with the BOS token)
			//System.out.println("Decoder In:");
			//shift(trainingSet[i][1], BOS).print();
			
			for(int j = 0; j < encoders.length; j++)//For each encoder/decoder in the stack,
			{
				//System.out.println("Layer " + j);
				encoders[j].activation(null, false);//Activates the encoder
				decoders[j].activation(null, false);//Activates the decoder (this must be after the encoder because the decoders depend on the encoders)
			}
			
			//Activates the output stack.
			rotate.activation(null, false);
			
			SimpleMatrix modelOut;
			
			for(int j = 0; j < decoderVocabSizes.length; j++)
			{
				outputLinears[j].activation(null, false);
				modelOut = outputs[j].activation(null, false);
				
				costSum += cost.cost(modelOut, trainingSet[i][1].getColumn(j));
				
				SimpleMatrix lastErrorIn = cost.derivative(modelOut, trainingSet[i][1].getColumn(j));
				
				outputs[j].reportGradient(lastErrorIn);
			}
			
			//System.out.println("===========BACK============ ");
			
			for(int j = model.length - 1; j >= 0; j--)//For each layer in the model,
			{
				model[j].backprop();//Backprops that layer.
			}
			//int s = 0/0;
		}
		
		return costSum / (trainingSet.length * decoderVocabSizes.length);
	}
	
	/**
	 * Runs inference on the given input. Uses greedy decoding to get the outputs. Use {@link #beamSearch(SimpleMatrix, int)} for
	 * a non-greedy inference.
	 * @param input The input to run inference on.
	 * @return A sequence of tokens. Note: this has length decoderSequenceLength + 1 because it includes the BOS token.
	 */
	@Override
	public SimpleMatrix feedForward(SimpleMatrix input) {
		encoderIn.activation(input, true);//Activates the encoder input layer with the given input
		
		//System.out.println("Inferencing:");
		//input.print();
		
		for(int i = 0; i < encoders.length; i++)//Activates all of the encoders
		{
			encoders[i].activation(null, true);
		}
		
		/*
		 * This is the input for the decoder. Starting out, it is just a BOS token and a bunch of padding tokens.
		 * It is fed forward through the decoder, and the we grab the next token from the model, add it to the decoder
		 * input, re-run the decoder, and grab the next token.
		 */
		SimpleMatrix currentIn = SimpleMatrix.filled(decoderSequenceLength+1, decoderVocabSizes.length, PADDING);
		currentIn.setRow(0, SimpleMatrix.filled(1, decoderVocabSizes.length, BOS));
		
		for(int i = 0; i < decoderSequenceLength; i++)//For each item in the decoder sequence,
		{
			//System.out.println("\nDecoder input:");
			//currentIn.print();
			decoderIn.activation(currentIn, true);//Activates the decoder input with the current decoder input
			for(int j = 0; j < decoders.length; j++)//For each decoder,
			{
				decoders[j].activation(null, true);//Activate the decoder
			}
			
			rotate.activation(null, true);
			
			for(int j = 0; j < decoderVocabSizes.length; j++)
			{
				//Activate the output stack
				outputLinears[j].activation(null, true);
				outputs[j].activation(null, true);
				
				//System.out.println("\nDecoder Output:");
				//output.getLastActivation().print();
				
				int newToken = NetworkMath.argmax(outputs[j].getLastActivation().getRow(i));//Gets the next token
				currentIn.set(i+1, j, newToken);//Add the token to the decoder input
			}
		}//Repeat
		return currentIn;
	}
	
//	/**
//	 * Testing method. Prints each encoder and decoder input in the training set. Then, runs inference on it. Because it's given
//	 * all the correct tokens as input, it doesn't use any special decoding method - it just takes all of the outputs of the decoder.
//	 * @param trainingSet The dataset to test. 
//	 */
//	public void test(SimpleMatrix[]... trainingSet)
//	{
//		System.out.println("===============TESTING===============");
//		float costSum = 0;
//		for(int i = 0; i < trainingSet.length; i++)//For each item in the dataset,
//		{
//			encoderIn.activation(trainingSet[i][0], true);//Activates the encoder input with the item's input
//			decoderIn.activation(shift(trainingSet[i][1], BOS), true);//Activates the decoder input with the target output
//			
//			//SimpleMatrix in = SimpleMatrix.filled(trainingSet[i][1].getNumRows(), 1, -1);
//			//in.set(0, BOS);
//			
//			//decoderIn.activation(in);
//			
//			System.out.println("\nEncoder in:");
//			trainingSet[i][0].print();//Prints the input to the encoder
//			
//			System.out.println("\nDecoder in:");
//			//in.print();
//			shift(trainingSet[i][1], BOS).print();//Prints the input to the decoder
//			
//			for(int j = 0; j < encoders.length; j++)//For each encoder/decoder in the stack,
//			{
//				encoders[j].activation(null, true);//Activates them
//				decoders[j].activation(null, true);
//			}
//			
//			rotate.activation(null, true);//Activates the output stack
//			outputLinear.activation(null, true);
//			output.activation(null, true);
//			
//			SimpleMatrix modelOut = output.getLastActivation();//Gets the model's output
//			
//			System.out.println("\nModel Out:");
//			modelOut.print();//Prints the model's output
//			
//			System.out.print("Cost: ");
//			
//			double costVal = cost.cost(modelOut, trainingSet[i][1]);
//			
//			System.out.println(costVal);//Prints the cost of this item
//			
//			costSum += costVal;
//		}
//		
//		System.out.println("\nTotal Cost: " + costSum / trainingSet.length);//Prints the total cost.
//	}

	/**
	 * Inferences the model on the given input using beam search.
	 * @param input The input to run inference on
	 * @param width The beam width
	 * @return The output, according to beam search.
	 */
	public SimpleMatrix beamSearch(SimpleMatrix input, int width)
	{
		//This implementation uses log probabilities so that the probabilities don't go to zero when you have long sequences. 
		
		encoderIn.activation(input, true);//Activates the encoder input
		
		for(int i = 0; i < encoders.length; i++)//For each encoder in the encoder stack,
		{
			encoders[i].activation(null, true);//Activates the encoder.
		}
		
		SimpleMatrix currentIn = SimpleMatrix.filled(decoderSequenceLength+1, decoderVocabSizes.length, PADDING);//The current input of the decoder, starts as just a BOS token and a bunch of padding
		currentIn.setRow(0, SimpleMatrix.filled(1, decoderVocabSizes.length, BOS));
		
		SimpleMatrix[] currentBest = new SimpleMatrix[width];//Matrix to hold the best options so far
		Double[] currentProbabilities = new Double[width];//Matrix to hold the probabilities of each of the best options (Double b/c you can't make a list of double)
		SimpleMatrix out;
		
		ArrayList<SimpleMatrix> best = new ArrayList<>();//A running list of the current best items
		ArrayList<Double> topProbabilities = new ArrayList<>();//A running list of the probabilities of each item
		
		Arrays.fill(currentProbabilities, 0d);
		Arrays.fill(currentBest, currentIn);
		
		for(int i = 1; i < decoderSequenceLength + 1; i++)//For each token after the first (position 0 is BOS and 1 is the first),
		{
			for(int d = 0; d < decoderVocabSizes.length; d++)
			{
				//System.out.println(best);
				best.clear();//Clear the list of best sequences
				topProbabilities.clear();//Clear the list of probabilities
				best.add(null);//Start best with null (The lists need at least one item )
				topProbabilities.add(Double.NEGATIVE_INFINITY);//Start the probabilities with negative infinity
				
				for(int j = 0; j < width; j++)//For each sequence j in the beam width (the sequences we are building off of),
				{
					if(currentBest[j] == null)//If the sequence is undefined
						continue;//Move on to the next width
					out = doDecoder(currentBest[j], true)[d];//Run the decoder on the sequence
					//out.print();
					
					for(int k = 0; k < decoderVocabSizes[d]; k++)//For each token in the voacb,
					{
						double probability = Math.log(out.get(i-1, k)) + currentProbabilities[j];//The probability is the probability of the sequence we're building off + the probability of the current token
						
						for(int l = 0; l < width; l++)//For each l in the width (of the list of new sequences we're building)
						{
							if(probability > topProbabilities.get(l) || topProbabilities.get(l) == Double.NEGATIVE_INFINITY)//If this probability is greater than the probability of the sequence at l
							{
								topProbabilities.add(l, probability);//Add the probability to the list at l
								SimpleMatrix newArray = currentBest[j].copy();//The new array (the old sequence of tokens + the new token)
								newArray.set(i, d, k);//Set the token at in the new token sequence to k
								best.add(l, newArray);//Add the new sequence to the list at l
								if(topProbabilities.size() > width)//If the number of sequences we're storing is greater than the beam width,
								{
									best.remove(topProbabilities.size() - 1);//Remove the last sequence
									topProbabilities.remove(topProbabilities.size() - 1);//Remove the last probability
								}
								break;//Move on to the next token.
							}
						}
					}
				}
				
				currentBest = best.toArray(currentBest);//Put the best list into the best array
				currentProbabilities = topProbabilities.toArray(currentProbabilities);//Same with the probabilities
			}
		}
		
		return currentBest[0];//Return the sequence with the best probability.
	}
	
	/**
	 * Activates the decoder stack on the given input.
	 * @param input The input to activate on.
	 * @param isInference Whether this is inference.
	 * @return The output of the model.
	 */
	private SimpleMatrix[] doDecoder(SimpleMatrix input, boolean isInference)
	{
		decoderIn.activation(input, isInference);
		for(int i = 0; i < decoders.length; i++)
			decoders[i].activation(null, isInference);
		rotate.activation(null, isInference);
		
		SimpleMatrix[] out = new SimpleMatrix[decoderVocabSizes.length];
		
		for(int j = 0; j < decoderVocabSizes.length; j++)
		{
			outputLinears[j].activation(null, isInference);
			out[j] = outputs[j].activation(null, isInference);
		}
		return out;
	}
	
	/**
	 * Shifts the given input right by the BOS token. Truncates the last token.
	 * @param input The input to shift.
	 * @param bosToken The BOS token.
	 * @return The input, shifted right
	 */
	public static SimpleMatrix shift(SimpleMatrix input, int bosToken)
	{
		input = input.copy();//Copies input because it uses the set method and we don't want to edit the input.
		
		//input.print();
		
		for(int j = 0; j < input.getNumCols(); j++)
		{
			for(int i = input.getNumRows() - 1; i >= 1; i--)//For each token from the last to the second-to-first
			{
				input.set(i, j, input.get(i-1, j));//Set the item at i to the one at i-1
			}
			
			input.set(0, j, bosToken);//Set the first one to the BOS token.
		}
		
		//input.print();
		
		return input;
	}
	
	/**
	 * Turns the model into a string that can be reconstructed by the {@link #load(String)} model.
	 */
	@Override
	public String stringify() {
		StringBuilder out = new StringBuilder("Transformer " + rotate.getId() + " " + encoderSequenceLength + " " + decoderSequenceLength + " " + embeddingDepth + " " + 
					 + heads + " " + decoders.length + " ");
		out.append(encoderVocabSizes.length);
		out.append(" ");
		for(int i = 0; i < encoderVocabSizes.length; i++)
		{
			out.append(encoderVocabSizes[i]);
			out.append(" ");
		}
		
		out.append(decoderVocabSizes.length);
		out.append(" ");
		for(int i = 0; i < decoderVocabSizes.length; i++)
		{
			out.append(decoderVocabSizes[i]);
			out.append(" ");
		}
		
		out.append(encoderIn.className());
		out.append("\n||");
		out.append(encoderIn.stringify());
		out.append("\n||");
		out.append(decoderIn.className());
		out.append("\n||");
		out.append(decoderIn.stringify());
		out.append("\n||");
		for(int i = 0; i < encoders.length; i++)
		{
			out.append(encoders[i].className());
			out.append("\n||");
			out.append(encoders[i].stringify());
			out.append("\n||");
			out.append(decoders[i].className());
			out.append("\n||");
			out.append(decoders[i].stringify());
			out.append("\n||");
		}
		
		for(int i = 0; i < decoderVocabSizes.length; i++)
		{
			out.append(outputLinears[i].className());
			out.append("\n||");
			out.append(outputLinears[i].stringify());
			out.append("\n||");
		}
		return out.toString();
	}
	
	public static TransformerModel2D load(String string)
	{
		Scanner scanner = new Scanner(string);
		if(!scanner.next().equals("Transformer"))
		{
			scanner.close();
			throw new IllegalArgumentException("Model type is not transformer!");
		}
		
		int rotateID = scanner.nextInt();
		int encoderSequenceLength = scanner.nextInt();
		int decoderSequenceLength = scanner.nextInt();
		int embedDepth = scanner.nextInt();
		int heads = scanner.nextInt();
		int layers = scanner.nextInt();
		
		int encoderDepth = scanner.nextInt();
		int[] encoderVocabSizes = new int[encoderDepth];
		for(int i = 0; i < encoderDepth; i++)
		{
			encoderVocabSizes[i] = scanner.nextInt();
		}
		
		int decoderDepth = scanner.nextInt();
		int[] decoderVocabSizes = new int[decoderDepth];
		for(int i = 0; i < decoderDepth; i++)
		{
			decoderVocabSizes[i] = scanner.nextInt();
		}
		
		TransformerModel2D model = new TransformerModel2D(encoderSequenceLength, decoderSequenceLength, embedDepth, encoderVocabSizes, decoderVocabSizes, heads, layers);
		
		scanner.useDelimiter("\\|\\|");
		
		scanner.next();//Clear class name
		model.encoderIn = TransformerInput.load(scanner.next(), model, 0);
		model.model[0] = model.encoderIn;
		model.reportLayer(model.model[0]);
		
		scanner.next();//Clear class name
		model.decoderIn = TransformerInput.load(scanner.next(), model, 1);
		model.model[1] = model.decoderIn;
		model.reportLayer(model.model[1]);
		
		for(int i = 0; i < layers; i++)
		{
			scanner.next();//Clear class name
			model.encoders[i] = Encoder.load(scanner.next(), model, 2*i+2);
			model.model[2*i+2] = model.encoders[i];
			model.reportLayer(model.model[2*i+2]);
			
			scanner.next();//Clear class name
			model.decoders[i] = Decoder.load(scanner.next(), model, 2*i+3);
			model.model[2*i+3] = model.decoders[i];
			model.reportLayer(model.model[2*i+3]);
		}
		
		model.rotate = new RotationLayer(model.decoders[layers-1]);
		model.model[2*layers+2] = model.rotate;
		model.rotate.setId(rotateID);
		model.reportLayer(model.rotate);
		
		for(int i = 0; i < decoderDepth; i++)
		{
			scanner.next();//Clear class name
			model.outputLinears[i] = StandardLayer.load(scanner.next(), model, 2*layers+3);
			model.model[2*layers+3+2*i] = model.outputLinears[i];
			model.reportLayer(model.outputLinears[i]);
			
			model.outputs[i] = new RotationLayer(model.outputLinears[i]);
			model.model[2*layers+4+2*i] = model.outputs[i];
			model.reportLayer(model.outputs[i]);
		}
		
		for(Layer layer : model.model)
			layer.setModel(model);
		
		scanner.close();
		return model;
	}
	
	@Override
	public String toString() {
		String out = "Model: (" + encoderSequenceLength + ", " + 1 + ") -> (" + decoderSequenceLength + ", " + decoderVocabSizes + ")\n";
		for(int i = 0; i < model.length; i++)
		{
			out += model[i] + "\n";
		}
		return out;
	}
}
