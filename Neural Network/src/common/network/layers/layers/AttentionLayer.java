package common.network.layers.layers;

import java.util.Scanner;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations.ElementOpReal;
import common.network.layers.Activation;
import common.network.layers.models.LayersNetwork;

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

	protected AttentionLayer() {super(0, 0);};
	
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
		
		oneOverSqrtKeyLen = (1/Math.sqrt(keySource.outputs));
	}

	//@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		masks = querySource.getMasks();
		valueLinear.activation(null);
		keyLinear.activation(null);
		queryLinear.activation(null);
		
		SimpleMatrix valueActivation = valueLinear.getLastActivation();
		SimpleMatrix keyActivation = keyLinear.getLastActivation();
		SimpleMatrix queryActivation = queryLinear.getLastActivation();

		for(int i = 0; i < heads; i++)
		{			
			SimpleMatrix valueData = valueActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			SimpleMatrix keyData = keyActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			SimpleMatrix queryData = queryActivation.cols(i * headDataSize, i * headDataSize + headDataSize);
			
			if(masking)
			{
				lastSoftIn[i] = mask(queryData.mult(keyData.transpose()), querySource, keySource, decoder).scale(oneOverSqrtKeyLen);
			}else {
				lastSoftIn[i] = queryData.mult(keyData.transpose()).scale(oneOverSqrtKeyLen);
			}			
			//lastSoftIn[i].print();
			lastSoftOut[i] = Activation.SOFTMAX_DEPTHWISE.activation(lastSoftIn[i]);
			
			SimpleMatrix attention = lastSoftOut[i].mult(valueData);
			
			lastActivation.insertIntoThis(0, i*headDataSize, attention);
		}
		
		
		return lastActivation;
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
			
			SimpleMatrix[] error = errorMatrixMult(lastSoftOut[i], valueData, nextErrorData);
			
			SimpleMatrix error2 = Activation.SOFTMAX_DEPTHWISE.error(lastSoftIn[i], error[0]);
			
			if(masking)
			{
				//error2 = maskBackProp(error2, querySource, keySource, decoder);
			}
			
			error2 = error2.scale(oneOverSqrtKeyLen);
			
			SimpleMatrix[] error3 = errorMatrixMultBT(queryData, keyData, error2);
			
			valueError.insertIntoThis(0, i*headDataSize, error[1]);
			keyError.insertIntoThis(0, i*headDataSize, error3[1]);
			queryError.insertIntoThis(0, i*headDataSize, error3[0]);
		}
		
		valueLinear.reportGradient(valueError);
		keyLinear.reportGradient(keyError);
		queryLinear.reportGradient(queryError);
		valueLinear.backprop();
		queryLinear.backprop();
		keyLinear.backprop();
	}
	
	static SimpleMatrix mask(SimpleMatrix matrix, Layer querySource, Layer keySource, boolean isDecoder)
	{
		//isDecoder = true;
		if(isDecoder)
		{
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return //querySource.getMasks()[row] || keySource.getMasks()[col] ||
							col > row ? Double.NEGATIVE_INFINITY : value;
				}
			});
		}else {return matrix;/*
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return querySource.getMasks()[row] || keySource.getMasks()[col] ? Double.NEGATIVE_INFINITY : value;
				}
			});//*/
		}
	}
	
	static SimpleMatrix maskBackProp(SimpleMatrix matrix, Layer querySource, Layer keySource, boolean isDecoder)
	{
		if(isDecoder)
		{
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return col > row || querySource.getMasks()[row] || keySource.getMasks()[col] ? 0 : value;
				}
			});
		}else {
			return matrix.elementOp(new ElementOpReal() {
				
				@Override
				public double op(int row, int col, double value) {
					return querySource.getMasks()[row] || keySource.getMasks()[col] ? 0 : value;
				}
			});
		}
	}
	
	static SimpleMatrix[] errorMatrixMult(SimpleMatrix a, SimpleMatrix b, SimpleMatrix error) 
	{
		SimpleMatrix aError = error.mult(b.transpose());
		SimpleMatrix bError = error.transpose().mult(a).transpose();
		
		return new SimpleMatrix[] {aError,bError};
	}
	
	static SimpleMatrix[] errorMatrixMultBT(SimpleMatrix a, SimpleMatrix b, SimpleMatrix error)
	{
		SimpleMatrix aError = error.mult(b);
		SimpleMatrix bError = error.transpose().mult(a);
		
		return new SimpleMatrix[] {aError, bError};
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
	
	@Override
	public String stringify() {
		return getId() + " " + valueSource.getId() + " " + keySource.getId() + " " + querySource.getId() + " " + heads + " " + masking + " " + decoder + " "
		+ "\n" + valueLinear.stringify() + "\n$$\n" + keyLinear.stringify() + "\n$$\n" + queryLinear.stringify() + "\n$$\n";
	}

	@Override
	public AttentionLayer load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int valueID = scanner.nextInt();
		int keyID = scanner.nextInt();
		int queryID = scanner.nextInt();
		int heads = scanner.nextInt();
		boolean masking = scanner.nextBoolean();
		boolean decoder = scanner.nextBoolean();
		AttentionLayer out = new AttentionLayer(model.getLayerByID(valueID), model.getLayerByID(keyID), model.getLayerByID(queryID), heads, masking, decoder);
		out.setId(id);
		scanner.useDelimiter("$$");
		out.valueLinear = out.keyLinear.load(scanner.next(), model, position);
		out.keyLinear = out.keyLinear.load(scanner.next(), model, position);
		out.queryLinear = out.keyLinear.load(scanner.next(), model, position);
		scanner.close();
		return out;
	}
}
