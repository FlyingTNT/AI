package common.network.layers.layers;

import common.network.layers.Activation;
import common.network.layers.LayersMain;
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
	
	float oneOverSqrtKeyLen;
	
	float[][][] lastSoftIn;
	float[][][] lastSoftOut;
	
	public AttentionLayer(Layer valueSource, Layer keySource, Layer querySource, int heads, boolean masking, boolean decoder) {
		super(querySource.outputs, querySource.outputs);
		depth = querySource.depth;
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
		lastSoftIn = new float[heads][][];
		lastSoftOut = new float[heads][][];
		lastActivation = new float[outputs][depth];
		
		oneOverSqrtKeyLen = (float)(1/Math.sqrt(keySource.outputs));
	}

	@Override
	public float[][] activation(float[][] input) {
		float[][] valueActivation = valueLinear.activation(valueSource.getLastActivation());
		float[][] keyActivation = keyLinear.activation(keySource.getLastActivation());
		float[][] queryActivation = queryLinear.activation(querySource.getLastActivation());
		//System.out.println("vkq");
		//System.out.println(LayersMain.floatMatrixToString(valueActivation, 2));
		//System.out.println(LayersMain.floatMatrixToString(keyActivation, 2));
		//System.out.println(LayersMain.floatMatrixToString(queryActivation, 2));
		
		//System.out.println("VKQ:");
		//LayersMain.print(valueActivation);
		//LayersMain.print(keyActivation);
		//LayersMain.print(queryActivation);
		
		int embDepth = 0;
		int embDepth2 = 0;
		for(int i = 0; i < heads; i++)
		{
			float[][] valueData = new float[valueSource.outputs][headDataSize];
			float[][] keyData = new float[keySource.outputs][headDataSize];
			float[][] queryData = new float[querySource.outputs][headDataSize];
			
			for(int j = 0; j < headDataSize; j++)
			{
				for(int k = 0; k < valueSource.outputs; k++)
				{
					valueData[k][j] = valueActivation[k][embDepth];
					keyData[k][j] = keyActivation[k][embDepth];
					queryData[k][j] = queryActivation[k][embDepth];
				}
				embDepth++;
			}
			
			if(masking)
			{
				lastSoftIn[i] = mask(NetworkMath.scale(NetworkMath.multiplyABT(queryData, keyData), oneOverSqrtKeyLen), querySource, keySource, decoder);
			}else {
				lastSoftIn[i] = NetworkMath.scale(NetworkMath.multiplyABT(queryData, keyData), oneOverSqrtKeyLen);
			}
			lastSoftOut[i] = Activation.SOFTMAX_DEPTHWISE.activation(lastSoftIn[i]);
			
			/*System.out.println("SoftI/O:");
			LayersMain.print(lastSoftIn[i]);
			LayersMain.print(lastSoftOut[i]);*/
			
			float[][] attention = NetworkMath.multiplyAB(lastSoftOut[i], valueData);
			
			//System.out.println(LayersMain.floatMatrixToString(lastSoftIn[0], 2));
			
			for(int j = 0; j < headDataSize; j++)
			{
				for(int k = 0; k < outputs; k++)
				{
					lastActivation[k][embDepth2] = attention[k][j];
				}
				embDepth2++;
			}
		}
		return lastActivation;
	}

	@Override
	public void backprop() {
		float[][] nextErrorWeighted = getGradient();
		clearGradients();
		int embDepth = 0;
		int embDepth2 = 0;
		
		float[][] valueError = new float[valueSource.outputs][depth];
		float[][] keyError = new float[keySource.outputs][depth];
		float[][] queryError = new float[querySource.outputs][depth];
		
		for(int i = 0; i < heads; i++)
		{
			float[][] valueData = new float[valueSource.outputs][headDataSize];
			float[][] keyData = new float[keySource.outputs][headDataSize];
			float[][] queryData = new float[querySource.outputs][headDataSize];
			float[][] nextErrorData = new float[querySource.outputs][headDataSize];
			
			for(int j = 0; j < headDataSize; j++)
			{
				for(int k = 0; k < valueSource.outputs; k++)
				{
					valueData[k][j] = valueLinear.getLastActivation()[k][embDepth];
					keyData[k][j] = keyLinear.getLastActivation()[k][embDepth];
					queryData[k][j] = queryLinear.getLastActivation()[k][embDepth];
					nextErrorData[k][j] = nextErrorWeighted[k][embDepth];
				}
				embDepth++;
			}
			
			float[][][] error = errorMatrixMult(lastSoftOut[i], valueData, nextErrorData);
			
			float[][] error2 = Activation.SOFTMAX_DEPTHWISE.error(lastSoftIn[i], error[0]);
			
			///*
			if(masking)
			{
				maskBackProp(error2, querySource, keySource, decoder);
			}
			//*/
			
			NetworkMath.scale(error2, oneOverSqrtKeyLen);
			
			float[][][] error3 = errorMatrixMultBT(queryData, keyData, error2);
			
			for(int j = 0; j < headDataSize; j++)
			{
				for(int k = 0; k < outputs; k++)
				{
					valueError[k][embDepth2] = error[1][k][j];
					keyError[k][embDepth2] = error3[1][k][j];
					queryError[k][embDepth2] = error3[0][k][j];
				}
				embDepth2++;
			}
		}
		//System.out.println("vkq:");
		//System.out.println(LayersMain.floatMatrixToString(valueError, 1));
		//System.out.println(LayersMain.floatMatrixToString(keyError, 1));
		//System.out.println(LayersMain.floatMatrixToString(queryError, 1));
		
		valueLinear.reportGradient(valueError);
		keyLinear.reportGradient(keyError);
		queryLinear.reportGradient(queryError);
		valueLinear.backprop();
		queryLinear.backprop();
		keyLinear.backprop();
	}
	
	static float[][] attention(float[][] query, float[][] key, float[][] value)
	{
		return NetworkMath.multiplyAB(Activation.SOFTMAX_DEPTHWISE.activation(NetworkMath.scale(NetworkMath.multiplyABT(query, key), 1/(float)Math.sqrt(key.length))), value);
	}
	
	static float[][] mask(float[][] matrix, Layer querySource, Layer keySource, boolean isDecoder)
	{
		for(int i = 0; i < matrix.length; i++)
		{
			for(int j = 0; j < matrix[0].length; j++)
			{
				matrix[i][j] = j > i ? Float.NEGATIVE_INFINITY : matrix[i][j];
			}
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
