package common.network.layers.layers;

import java.util.Scanner;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;
import common.network.layers.Activation;
import common.network.layers.models.LayersModel;

/**
 * A multi-head attention layer, as defined in Google's Attention is All You Need.
 * @author C. Cooper
 */
public class AttentionLayer extends Layer {
	Layer valueSource;//The layer the value inputs come from.
	Layer keySource;//The layer the key inputs come from.
	Layer querySource;//The layer the query inputs come from.
	
	StandardLayer valueLinear;//The linear layer to apply to the values
	StandardLayer keyLinear;//The linear layer to apply to the keys
	StandardLayer queryLinear;//The linear layer to apply to the queries
	
	int heads;//The number of heads
	int headDataSize;//The amount of the embed depth each head deals with
	
	boolean masking;//Whether to do masking
	boolean isDecoder;//Whether this is a decoder (decoders don't mask padding during inference, and encoder does)
	boolean casualMask;//Whether to apply the casual mask
	
	double oneOverSqrtKeyLen;//1/(sqrt(# of outputs of key source))
	
	SimpleMatrix[] lastSoftIn;//The last input to the softmax part of the attention
	SimpleMatrix[] lastSoftOut;//The last output of the softmax part of the attention

	/**
	 * Barebones constructor used for the {@link #load(String, LayersModel, int)} method.
	 */
	protected AttentionLayer() {super(0, 0);};
	
	/**
	 * Basic constructor.
	 * @param valueSource The layer the value info comes from.
	 * @param keySource The layer the key info comes from.
	 * @param querySource The layer the query info comes from.
	 * @param heads The number of heads to use.
	 * @param useCasualMask Whether to apply the casual mask.
	 * @param masking Whether to do masking.
	 * @param isDecoder Whether this is a decoder.
	 */
	public AttentionLayer(Layer valueSource, Layer keySource, Layer querySource, int heads, boolean useCasualMask, boolean masking, boolean isDecoder) {
		super(querySource.outputs, querySource.outputs);
		depth = querySource.depth;
		setGradientSize(outputs, depth);
		this.valueSource = valueSource;
		this.keySource = keySource;
		this.querySource = querySource;
		this.valueLinear = new StandardLayer(valueSource, valueSource.outputs, Activation.NONE);
		this.keyLinear = new StandardLayer(keySource, keySource.outputs, Activation.NONE);
		this.queryLinear = new StandardLayer(querySource, querySource.outputs, Activation.NONE);
		this.masking = masking;
		this.isDecoder = isDecoder;
		this.casualMask = useCasualMask;
		
		this.heads = heads;
		if(depth/heads*heads != depth)
		{
			throw new IllegalArgumentException("Embedding depth must be a multiple of the head count!");
		}
		headDataSize = depth/heads;
		lastSoftIn = new SimpleMatrix[heads];
		lastSoftOut = new SimpleMatrix[heads];
		lastActivation = new SimpleMatrix(outputs, depth);
		
		oneOverSqrtKeyLen = (1/Math.sqrt(keySource.outputs));
	}

	/**
	 * Does the activation of this layer.
	 * <br><br>
	 * Activation = softmax(Q(K^T))V (Q, K, and V are 2d matrices)
	 * @param input Ignored.
	 * @param isInference Whether this is running inference (as opposed to training).
	 * @return This layer's activation
	 */
	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		/*
		 * You can also look at this link for an explanation: https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
		 * I used that website a lot when making this. Its visuals for that article kinda suck imo though.
		 * 
		 * An attention layer gets its data from three places: a key source, a value source, and a query source. In a self attention
		 * layer, these are all the same. In a transformer, generally the encoders only have self attention, while the decoders
		 * have one self attention layer and one attention layer whose queries are the decoder's self attention's output, and whose
		 * keys and values come from the decoder's corresponding encoder. These inputs are matrices of size sequence length x embed depth.
		 * The sequence length may be different between the encoders and decoders, but the embed depth must be the same. In multi-head
		 * attention, there are multiple heads that each get a section of the input, so that each head learns to focus on a certain aspect.
		 * Each of the inputs are then passed through a standard linear layer and then operated upon. There is a separate layer for the key,
		 * value and query.
		 * The inputs are split along the embed axis as so:
		 * Input (sequence len = 5 x embed depth = 8):
		 * [1 0 0 0 0 0 0 0]
		 * [1 1 0 0 0 0 1 0]
		 * [0 0 0 0 1 0 1 1]
		 * [0 0 1 0 1 1 0 0]
		 * [0 0 1 1 0 0 0 0]
		 * Split by 4 heads:
		 * [1 0] [0 0] [0 0] [0 0]
		 * [1 1] [0 0] [0 0] [1 0]
		 * [0 0] [0 0] [1 0] [1 1]
		 * [0 0] [1 0] [1 1] [0 0]
		 * [0 0] [1 1] [0 0] [0 0]
		 * The first head would take the first matrix, and the second the second...
		 * Each head would then apply the attention equation to their data (see the equation above).
		 * Then each head's output would be re-combined (in the same way it was split) 
		 */
		
		masks = querySource.getMasks();//Pulls the query source's masks forward.
		SimpleMatrix valueActivation = valueLinear.activation(null, isInference);//Activates the value linear layer
		SimpleMatrix keyActivation = keyLinear.activation(null, isInference);//Activates the key linear layer
		SimpleMatrix queryActivation = queryLinear.activation(null, isInference);//Activates the query linear layer

		for(int i = 0; i < heads; i++)//For each head,
		{			
			//Gets the columns of the input that belong to this head
			SimpleMatrix valueData = valueActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			SimpleMatrix keyData = keyActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			SimpleMatrix queryData = queryActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			
			/*
			 * Performs the Q(K^T) part of the activation and stores it in lastSoftIn. If we're doing masking, also masks this data.
			 * This involves masking padding tokens and applying the casual mask if necessary.
			 */
			if(masking)
			{
				lastSoftIn[i] = mask(queryData.mult(keyData.transpose()), querySource.getMasks(), keySource.getMasks(), isDecoder, casualMask, isInference).scale(oneOverSqrtKeyLen);
			}else {
				lastSoftIn[i] = queryData.mult(keyData.transpose()).scale(oneOverSqrtKeyLen);
			}			
			
			lastSoftOut[i] = Activation.SOFTMAX_DEPTHWISE.activation(lastSoftIn[i]);//Takes the softmax of the above (along the embed depth axis)
			
			SimpleMatrix attention = lastSoftOut[i].mult(valueData);//Multiplies the softmax output by the value data for this head.
			
			lastActivation.insertIntoThis(0, i*headDataSize, attention);//Inserts the output into the lastActivation matrix in this head's allotted space.
		}
		
		return lastActivation;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = new SimpleMatrix(getGradient());
		clearGradients();
		
		double len = nextErrorWeighted.normF();//Clips the gradient if its magnitude is over 1 (b/c this layer has matrix mult, it tends to cause exploding gradients)
		if(len > 1)
			nextErrorWeighted = nextErrorWeighted.divide(len);
		
		//Matrices for the errors that will be sent back to the value, key, and query layers.
		SimpleMatrix valueError = new SimpleMatrix(valueSource.outputs, depth);
		SimpleMatrix keyError = new SimpleMatrix(keySource.outputs, depth);
		SimpleMatrix queryError = new SimpleMatrix(querySource.outputs, depth);
		
		//Gets the activations from feed forward
		SimpleMatrix valueActivations = valueSource.getLastActivation();
		SimpleMatrix keyActivations = keySource.getLastActivation();
		SimpleMatrix queryActivations = querySource.getLastActivation();
		
		for(int i = 0; i < heads; i++)//For each head,
		{
			//Gets the section of data from the error and key, query, and value linears that pertains to this head.
			SimpleMatrix valueData = valueActivations.cols(i*headDataSize, i*headDataSize+headDataSize);
			SimpleMatrix keyData = keyActivations.cols(i*headDataSize, i*headDataSize+headDataSize);
			SimpleMatrix queryData = queryActivations.cols(i*headDataSize, i*headDataSize+headDataSize);
			SimpleMatrix nextErrorData = nextErrorWeighted.cols(i*headDataSize, i*headDataSize+headDataSize);
			
			/*
			 * Pulls the error through the matrix multiplication softmax(Q*(K^T)) * Value
			 * error[0] = the error of softmax(Q*(K^T))
			 * error[1] = the error of Value
			 */
			SimpleMatrix[] error = errorMatrixMult(lastSoftOut[i], valueData, nextErrorData);
			
			/*
			 * Pulls the error through the softmax function
			 * error2 = the error of (Q*(K^T))
			 */
			SimpleMatrix error2 = Activation.SOFTMAX_DEPTHWISE.error(lastSoftIn[i], error[0]);
			
			/*
			 * If we're doing masking, masks the gradients as necessary (so that we don't edit weights that were masked
			 * on the forward pass and thus didn't actually contribute to the error).
			 */
			if(masking)
			{
				error2 = maskBackProp(error2, querySource.getMasks(), keySource.getMasks(), casualMask);
			}
			
			/*
			 * Scales error2 by oneOverSqrtKeyLen (Q*(K^T) was scaled by this on the forward pass)
			 */
			error2 = error2.scale(oneOverSqrtKeyLen);
			
			/*
			 * Pulls error2 through the matrix multiplication (Q*(K^T))
			 * error3[0] = error of Q
			 * error3[1] = error of K (the function accounts for the ^T)
			 */
			SimpleMatrix[] error3 = errorMatrixMultBT(queryData, keyData, error2);
			
			/*
			 * Adds the errors we calculated to the total V, K, and Q errors.
			 */
			valueError.insertIntoThis(0, i*headDataSize, error[1]);
			keyError.insertIntoThis(0, i*headDataSize, error3[1]);
			queryError.insertIntoThis(0, i*headDataSize, error3[0]);
		}
		
		/*
		 * Sends the errors back to each of their layers and backprops them. (because they are sub-layers of this layer, the
		 * model doesn't know they exist and thus won't call their backprop functions)
		 */
		valueLinear.reportGradient(valueError);
		keyLinear.reportGradient(keyError);
		queryLinear.reportGradient(queryError);
		valueLinear.backprop();
		queryLinear.backprop();
		keyLinear.backprop();
	}
	
	/**
	 * Masks the given matrix.
	 * <br><br>
	 * Uses the query and key masks to know what to mask.
	 * For example, with 4x4 input and query masks = [F, F, T, T] and key masks = [F, F, F, T], this mask would look
	 * like: (T is masked, F is not)<br>
	 * [F F F T]<br>
	 * [F F F T]<br>
	 * [T T T T]<br>
	 * [T T T T]<br>
	 * Also adds the casual mask on top of this if called for.
	 * @param matrix The matrix to mask.
	 * @param queryMasks The tokens in the query source that were masked (the locations of the padding tokens)
	 * @param keyMasks The tokens in the key source that were masked (the locations of the padding tokens)
	 * @param isDecoder Whether the masking is for a decoder
	 * @param casualMask Whether to apply the casual mask
	 * @param isInference Whether this is for inference
	 * @return The input matrix but with -Infinity in place of all the masked values.
	 */
	static SimpleMatrix mask(SimpleMatrix matrix, boolean[] queryMasks, boolean[] keyMasks, boolean isDecoder, boolean casualMask, boolean isInference)
	{
		SimpleMatrix out = matrix;
		if(casualMask)//If it should apply the casual mask,
		{
			out = out.elementOp(new ElementOpReal() {
				@Override
				public double op(int row, int col, double value) {
					return col > row ? Double.NEGATIVE_INFINITY : value;
				}
			});
		}
		
		if(!isInference || (isInference && !isDecoder))//If it's training, or it's inferencing but not on a decoder
		{
			out = out.elementOp(new ElementOpReal() {
				@Override
				public double op(int row, int col, double value) {
					return queryMasks[row] || keyMasks[col] ? Double.NEGATIVE_INFINITY : value;
				}
			});
		}
		return out;
	}
	
	
	/**
	 * Masks the given matrix for backprop. Same as {@link #mask(SimpleMatrix, boolean[], boolean[], boolean, boolean, boolean) mask()}
	 * except that masks with 0 instead of -Infinity. Also, it has fewer arguments because it is only called during training.
	 * <br><br>
	 * Uses the query and key masks to know what to mask.
	 * For example, with 4x4 input and query masks = [F, F, T, T] and key masks = [F, F, F, T], this mask would look
	 * like: (T is masked, F is not)<br>
	 * [F F F T]<br>
	 * [F F F T]<br>
	 * [T T T T]<br>
	 * [T T T T]<br>
	 * Also adds the casual mask on top of this if called for.
	 * @param matrix The matrix to mask.
	 * @param queryMasks The tokens in the query source that were masked (the locations of the padding tokens)
	 * @param keyMasks The tokens in the key source that were masked (the locations of the padding tokens)
	 * @param casualMask Whether to apply the casual mask
	 * @return The input matrix but with 0 in place of all the masked values.
	 */
	static SimpleMatrix maskBackProp(SimpleMatrix matrix, boolean[] queryMasks, boolean[] keyMasks, boolean casualMasking)
	{
		if(casualMasking)//If casualMasking, applies the casual and padding masks.
		{
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return col > row || queryMasks[row] || keyMasks[col] ? 0 : value;
				}
			});
		}else {//Otherwise, just applies the padding masks.
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return queryMasks[row] || keyMasks[col] ? 0 : value;
				}
			});
		}
	}
	
	/**
	 * Pulls the given error back through the matrix multiplication of the given matrices.
	 * @param a The first matrix that was multiplied.
	 * @param b The second matrix that was multiplied.
	 * @param error The error of the multiplication.
	 * @return A SimpleMatrix array in the form: {aError, bError}.
	 */
	static SimpleMatrix[] errorMatrixMult(SimpleMatrix a, SimpleMatrix b, SimpleMatrix error) 
	{
		/*
		 * The equations for each error. I just figured these by working it out on paper. I'm not gonna explain why it is;
		 * if you know about Jacobians, you should be able to work it out for yourself if you really want to know. It's not
		 * nearly as hard as I thought it would be.
		 */
		SimpleMatrix aError = error.mult(b.transpose());
		SimpleMatrix bError = error.transpose().mult(a).transpose();
		
		return new SimpleMatrix[] {aError, bError};
	}
	
	/**
	 * Pulls the given error back through the matrix multiplication of the given matrices, except the second should be transposed.
	 * @param a The first matrix that was multiplied.
	 * @param b The second matrix that was multiplied (not the transpose of it).
	 * @param error The error of the multiplication.
	 * @return A SimpleMatrix array in the form: {aError, bError}.
	 */
	static SimpleMatrix[] errorMatrixMultBT(SimpleMatrix a, SimpleMatrix b, SimpleMatrix error)
	{
		/*
		 * Exactly the same as the above function except that everything with b is transposed.
		 */
		SimpleMatrix aError = error.mult(b);
		SimpleMatrix bError = error.transpose().mult(a);
		
		return new SimpleMatrix[] {aError, bError};
	}
	
	@Override
	public void setModel(LayersModel model) {
		super.setModel(model);
		queryLinear.setModel(model);
		valueLinear.setModel(model);
		keyLinear.setModel(model);
	}

	@Override
	public String name() {
		return "Attention";
	}
	
	@Override
	public String stringify() {
		/*
		 * Returns a string in the form:
		 * thisId valueSourceId keySourceId querySourceId numHeads doMasking isDecoder doCasualMask
		 * valueLinear.stringify()
		 * $$
		 * keyLinear.stringify()
		 * $$
		 * queryLinear.stringify()
		 * $$
		 */
		return getId() + " " + valueSource.getId() + " " + keySource.getId() + " " + querySource.getId() + " " + heads + " " + masking + " " + isDecoder + " "
		+ casualMask + "\n" + valueLinear.stringify() + "\n$$\n" + keyLinear.stringify() + "\n$$\n" + queryLinear.stringify() + "\n$$\n";
	}

	/**
	 * Loads an AttentionLayer based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static AttentionLayer load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();//Gets this id.
		int valueID = scanner.nextInt();//Gets the id of the value layer.
		int keyID = scanner.nextInt();//Gets the id of the key layer.
		int queryID = scanner.nextInt();//Gets the id of the query layer.
		int heads = scanner.nextInt();//Gets the number of heads.
		boolean masking = scanner.nextBoolean();//Gets whether to do masking.
		boolean decoder = scanner.nextBoolean();//Gets if this is in a decoder.
		boolean casualMask = scanner.nextBoolean();//Gets whether to apply the casual mask.
		//Makes a new AttentionLayer based on the info parsed so far
		AttentionLayer out = new AttentionLayer(model.getLayerByID(valueID), model.getLayerByID(keyID), model.getLayerByID(queryID), heads, casualMask, masking, decoder);
		out.setId(id);//Sets the id of the new layer to the parsed id.
		scanner.useDelimiter("\\$\\$");//Sets the delimiter to "$$" (the string used to separate the linear layers)
		out.valueLinear = StandardLayer.load(scanner.next(), model, position);//Loads the value linear layer
		out.keyLinear = StandardLayer.load(scanner.next(), model, position);//Loads the key linear layer
		out.queryLinear = StandardLayer.load(scanner.next(), model, position);//Loads the query linear layer
		scanner.close();
		return out;
	}
}
