package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;

import common.network.layers.Activation;
import common.network.layers.TransMain;
import common.network.layers.models.LayersNetwork;
import common.network.math.NetworkMath;

public class AttentionLayer extends Layer {
	
	Layer valueSource;
	Layer keySource;
	Layer querySource;
	
	StandardLayer valueLinear;
	StandardLayer keyLinear;
	StandardLayer queryLinear;
	
	int heads;
	int headDataSize;
	
	boolean masking;
	boolean decoder;
	
	double oneOverSqrtKeyLen;
	
	SimpleMatrix[] lastSoftIn;
	SimpleMatrix[] lastSoftOut;
	
	final SimpleMatrix casualMask;
	
	public AttentionLayer(Layer valueSource, Layer keySource, Layer querySource, int heads, boolean masking, boolean decoder) {
		super(querySource.outputs, querySource.outputs);
		depth = querySource.depth;
		setGradientSize(outputs, depth);
		querySource.nextLayer = this;
		this.valueSource = valueSource;
		this.keySource = keySource;
		this.querySource = querySource;
		this.valueLinear = new StandardLayer(valueSource, valueSource.outputs, Activation.NONE);
		this.keyLinear = new StandardLayer(keySource, keySource.outputs, Activation.NONE);
		this.queryLinear = new StandardLayer(querySource, querySource.outputs, Activation.NONE);
		this.masking = masking;
		this.decoder = decoder;
		
		this.heads = heads;
		if(depth/heads*heads != depth)
		{
			throw new IllegalArgumentException("Embedding depth must be a multiple of the head count!");
		}
		headDataSize = depth/heads;
		lastSoftIn = new SimpleMatrix[heads];
		lastSoftOut = new SimpleMatrix[heads];
		lastActivation = new SimpleMatrix(new float[outputs][depth]);
		
		casualMask = new SimpleMatrix(casualMask(querySource.outputs));
		
		oneOverSqrtKeyLen = (1/Math.sqrt(keySource.outputs));
	}

	//@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		valueLinear.activation(null);
		keyLinear.activation(null);
		queryLinear.activation(null);
		
		SimpleMatrix valueActivation = valueLinear.getLastActivation();
		SimpleMatrix keyActivation = keyLinear.getLastActivation();
		SimpleMatrix queryActivation = queryLinear.getLastActivation();
		//System.out.println("vkq");
		//System.out.println(LayersMain.floatMatrixToString(valueActivation, 2));
		//System.out.println(LayersMain.floatMatrixToString(keyActivation, 2));
		//System.out.println(LayersMain.floatMatrixToString(queryActivation, 2));
		
		//System.out.println("VKQ:");
		//LayersMain.print(valueActivation);
		//LayersMain.print(keyActivation);
		//LayersMain.print(queryActivation);

		for(int i = 0; i < heads; i++)
		{			
			SimpleMatrix valueData = valueActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			SimpleMatrix keyData = keyActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			SimpleMatrix queryData = queryActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			
			if(masking)
			{
				lastSoftIn[i] = mask(queryData.mult(keyData.transpose()).scale(oneOverSqrtKeyLen), querySource, keySource, decoder);
			}else {
				lastSoftIn[i] = queryData.mult(keyData.transpose()).scale(oneOverSqrtKeyLen);
			}			
			lastSoftOut[i] = Activation.SOFTMAX_DEPTHWISE.activation(lastSoftIn[i]);
			
			/*System.out.println("SoftI/O:");
			LayersMain.print(lastSoftIn[i]);
			LayersMain.print(lastSoftOut[i]);*/
			
			//System.out.println("Head: " + i);
			//System.out.println(lastSoftIn[i]);
			//System.out.println(lastSoftOut[i]);
			
			SimpleMatrix attention = lastSoftOut[i].mult(valueData);
			
			//System.out.println(LayersMain.floatMatrixToString(lastSoftIn[0], 2));
			
			lastActivation.insertIntoThis(0, i*headDataSize, attention);
		}
		
		return null;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = new SimpleMatrix(getGradient());
		clearGradients();
		
		double len = nextErrorWeighted.normF();
		if(len > 1)
			nextErrorWeighted = nextErrorWeighted.divide(len);
		
		SimpleMatrix valueError = new SimpleMatrix(new float[valueSource.outputs][depth]);
		SimpleMatrix keyError = new SimpleMatrix(new float[keySource.outputs][depth]);
		SimpleMatrix queryError = new SimpleMatrix(new float[querySource.outputs][depth]);
		
		SimpleMatrix valueActivations = new SimpleMatrix(valueSource.getLastActivation());
		SimpleMatrix keyActivations = new SimpleMatrix(keySource.getLastActivation());
		SimpleMatrix queryActivations = new SimpleMatrix(querySource.getLastActivation());
		
		for(int i = 0; i < heads; i++)
		{
			SimpleMatrix valueData = valueActivations.cols(i*headDataSize, i*headDataSize+headDataSize);
			SimpleMatrix keyData = keyActivations.cols(i*headDataSize, i*headDataSize+headDataSize);
			SimpleMatrix queryData = queryActivations.cols(i*headDataSize, i*headDataSize+headDataSize);
			SimpleMatrix nextErrorData = nextErrorWeighted.cols(i*headDataSize, i*headDataSize+headDataSize);
			
			
			//System.out.println("Error");
			SimpleMatrix[] error = errorMatrixMult(lastSoftOut[i], valueData, nextErrorData);
			//System.out.println(error[0]);
			//System.out.println(error[1]);
			
			//System.out.println(lastSoftIn[i]);
			SimpleMatrix error2 = Activation.SOFTMAX_DEPTHWISE.error(lastSoftIn[i], error[0]);
			//System.out.println("Error2");
			
			//System.out.println(lastSoftIn[i]);
			//System.out.println(lastSoftOut[i]);
			//System.out.println(error2);
			
			if(masking)
			{
				error2 = maskBackProp(error2, querySource, keySource, decoder&&masking);
			}
			
			error2 = error2.scale(oneOverSqrtKeyLen);
			
			//System.out.println("Error3");
			SimpleMatrix[] error3 = errorMatrixMultBT(queryData, keyData, error2);
			
			//System.out.println(error3[0]);
			//System.out.println(error3[1]);
			
			valueError.insertIntoThis(0, i*headDataSize, error[1]);
			keyError.insertIntoThis(0, i*headDataSize, error3[1]);
			queryError.insertIntoThis(0, i*headDataSize, error3[0]);
		}
		//System.out.println("vkq:");
		//System.out.println(LayersMain.floatMatrixToString(valueError, 1));
		//System.out.println(LayersMain.floatMatrixToString(keyError, 1));
		//System.out.println(LayersMain.floatMatrixToString(queryError, 1));
		
		//System.out.println("KVQ");
		//System.out.println(keyError);
		//System.out.println(valueError);
		//System.out.println(queryError);
		
		valueLinear.reportGradient(valueError);
		keyLinear.reportGradient(keyError);
		queryLinear.reportGradient(queryError);
		valueLinear.backprop();
		queryLinear.backprop();
		keyLinear.backprop();
	}
	
	SimpleMatrix mask(SimpleMatrix matrix, Layer querySource, Layer keySource, boolean isDecoder)
	{
		if(isDecoder)
		{
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return col > row ? Double.NEGATIVE_INFINITY : value;
				}
			});
		}
		/*if(isDecoder)
		{
			for(int i = 0; i < matrix.length; i++)
			{
				for(int j = 0; j < matrix[0].length; j++)
				{
					matrix[i][j] = j > i || querySource.masks[i][0] || keySource.masks[j][0] ? Float.NEGATIVE_INFINITY : matrix[i][j];
				}
			}
		}else {
			for(int i = 0; i < matrix.length; i++)
			{
				for(int j = 0; j < matrix[0].length; j++)
				{
					matrix[i][j] = querySource.masks[i][j] || keySource.masks[j][i] ? Float.NEGATIVE_INFINITY : matrix[i][j];
				}
			}
		}*/
		return matrix;
	}
	
	double[][] mask(double[][] matrix, Layer query, Layer key, boolean isDecoder)
	{
		if(isDecoder)
			for(int i = 0; i < matrix.length; i++)
			{
				for(int j = 0; j < matrix[0].length; j++)
				{
					matrix[i][j] = j > i ? Double.NEGATIVE_INFINITY : matrix[i][j];
				}
			}
		return matrix;
	}
	
	static float[][] maskBackProp(float[][] matrix, Layer querySource, Layer keySource, boolean isDecoder)
	{
		if(isDecoder)
		{
			for(int i = 0; i < matrix.length; i++)
			{
				for(int j = i+1; j < matrix[0].length; j++)
				{
					matrix[i][j] = 0;
				}
			}
		}else {
			for(int i = 0; i < matrix.length; i++)
			{
				for(int j = 0; j < matrix[0].length; j++)
				{
					matrix[i][j] = querySource.masks[i][j] || keySource.masks[j][i] ? 0 : matrix[i][j];
				}
			}
		}
		
		return matrix;
	}
	
	static SimpleMatrix maskBackProp(SimpleMatrix matrix, Layer querySource, Layer keySource, boolean isDecoder)
	{
		if(isDecoder)
		{
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return col > row ? 0 : value;
				}
			});
		}
		
		return matrix;
	}
	
	static SimpleMatrix[] errorMatrixMult(SimpleMatrix a, SimpleMatrix b, SimpleMatrix error) 
	{
		SimpleMatrix aError = error.mult(b.transpose());
		SimpleMatrix bError = error.transpose().mult(a).transpose();
		
		return new SimpleMatrix[] {aError,bError};
	}
	
	/*
	 * I just made this up, it may me wrong. I couldn't find anything online.
	 */
	static float[][][] errorMatrixMult(float[][] a, float[][] b, float[][] error)
	{
		if(a[0].length != b.length)
		{
			throw new IllegalArgumentException("The matrixes are the wrong size, dumbass.");
		}
		float[][][] out = new float[2][][];
		out[0] = new float[a.length][a[0].length];
		out[1] = new float[b.length][b[0].length];
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < b[0].length; j++)
			{
				for(int k = 0; k < b.length; k++)
				{
					out[0][i][k] += error[i][j] * b[k][j];
					out[1][k][j] += error[i][j] * a[i][k];
				}
			}
		}
		return out;
	}
	
	static SimpleMatrix[] errorMatrixMultBT(SimpleMatrix a, SimpleMatrix b, SimpleMatrix error)
	{
		SimpleMatrix aError = error.mult(b);
		SimpleMatrix bError = error.transpose().mult(a);
		
		return new SimpleMatrix[] {aError, bError};
	}
	
	static float[][][] errorMatrixMultBT(float[][] a, float[][] b, float[][] error)
	{
		if(a[0].length != b[0].length)
		{
			throw new IllegalArgumentException("The matrixes are the wrong size, dumbass.");
		}
		float[][][] out = new float[2][][];
		out[0] = new float[a.length][a[0].length];
		out[1] = new float[b.length][b[0].length];
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < b.length; j++)
			{
				for(int k = 0; k < a[0].length; k++)
				{
					out[0][i][k] += error[i][j] * b[j][k];
					out[1][j][k] += error[i][j] * a[i][k];
				}
			}
		}
		return out;
	}
	
	static double[][] casualMask(int sequenceLen)
	{
		double[][] out = new double[sequenceLen][sequenceLen];
		for(int i = 0; i < sequenceLen; i++)
		{
			for(int j = i+1; j < sequenceLen; j++)
			{
				out[i][j] = Double.NEGATIVE_INFINITY;
			}
		}	
		return out;
	}
	
	static SimpleMatrix softmaxDepthwiseError(SimpleMatrix activation, SimpleMatrix nextWeightedError)
	{
		SimpleMatrix softmax = activation;
		SimpleMatrix error = softmax.elementMult(nextWeightedError);
		
		float[][] out = new float[activation.getNumRows()][activation.getNumCols()];
		for(int w = 0; w < activation.getNumRows(); w++)
		for(int output = 0; output < activation.getNumCols(); output++)
		{
			for(int in = 0; in < activation.getNumCols(); in++)
			{
				//DERIVATIVE OF iTH OUTPUT WITH RESPECT TO jTH INPUT = Sig(i) * ((i==j?1:0) - Sig(j));
				if(in == output)//COMPUTING THE PARTIAL FOR EACH INPUT
				{
					out[w][in] += (1 - softmax.get(w, in)) * error.get(w, output);
				}else {
					out[w][in] += -softmax.get(w, in)* error.get(w, output);
				}
			}
		}
		
		return new SimpleMatrix(out);
	}
	
	@Override
	public void setModel(LayersNetwork model) {
		this.model = model;
		queryLinear.setModel(model);
		valueLinear.setModel(model);
		keyLinear.setModel(model);
	}
	
	public void setMasking(boolean masking) {
		this.masking = masking;
	}

	@Override
	public String name() {
		return "Attention";
	}

}
