package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersNetwork;

public class Encoder extends Layer{

	AttentionLayer attention;
	ResidualAddition attentionResidual;
	NormLayer attentionNorm;
	StandardLayer linear;
	ResidualAddition linearResidual;
	NormLayer linearNorm;
	
	Layer[] layers;

	public Encoder(Layer lastLayer, int heads, boolean masking) {
		super(lastLayer, lastLayer.outputs);
		this.depth = lastLayer.depth;
		
		attention = new AttentionLayer(lastLayer, lastLayer, lastLayer, heads, masking, false);
		attentionResidual = new ResidualAddition(attention, lastLayer);
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, lastLayer.outputs, Activation.RELU);
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		
		layers = new Layer[] {attention, attentionResidual, attentionNorm, linear, linearResidual, linearNorm};
	}
	
	private Encoder(Layer lastLayer, AttentionLayer attention, StandardLayer linear, ResidualAddition residualAddition, NormLayer norm) {
		super(lastLayer, lastLayer.outputs);
		this.depth = lastLayer.depth;
		
		this.attention = attention;
		attentionResidual = residualAddition;
		attentionNorm = norm;
		this.linear = linear;
		linearResidual = new ResidualAddition(linear, norm);
		linearNorm = new NormLayer(linearResidual);
		
		layers = new Layer[] {attention, residualAddition, norm, linear, linearResidual, linearNorm};
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		for(Layer layer : layers)
			layer.activation(null);
		return lastActivation;
	}

	@Override
	public void backprop() {
		linearNorm.backprop();
		linearResidual.backprop();
		linear.backprop();
		attentionNorm.backprop();
		attentionResidual.backprop();
		attention.backprop();
	}

	@Override
	public String name() {
		return "Encoder";
	}

	@Override
	public SimpleMatrix getLastActivation() {
		return linearNorm.getLastActivation();
	}
	
	@Override
	public void reportGradient(SimpleMatrix gradient) {
		linearNorm.reportGradient(gradient);
	}
	
	@Override
	public void setModel(LayersNetwork model) {
		super.setModel(model);
		for(Layer layer : layers)
			layer.setModel(model);
	}
	
	@Override
	public boolean[] getMasks() {
		return linearNorm.getMasks();
	}
	
	void setMasking(boolean masking)
	{
		attention.masking = masking;
	}
	
	@Override
	public String stringify() {
		StringBuilder builder = new StringBuilder();
		builder.append(getId() + " " + lastLayer.getId() + " "+ attentionNorm.getId() + "\n");
		builder.append(attention.stringify());
		builder.append("\n##\n");
		builder.append(linear.stringify());
		builder.append("\n##\n");
		return builder.toString();
	}
	
	public static Encoder load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		int normID = scanner.nextInt();
		scanner.useDelimiter("##");
		AttentionLayer attentionLayer = AttentionLayer.load(scanner.next(), model, position);
		model.reportLayer(attentionLayer);
		
		ResidualAddition residualAddition = new ResidualAddition(attentionLayer, model.getLayerByID(lastID));
		NormLayer norm = new NormLayer(residualAddition);
		norm.setId(normID);
		model.reportLayer(norm);		
		StandardLayer linear = StandardLayer.load(scanner.next(), model, position);
		scanner.close();
		Encoder out = new Encoder(model.getLayerByID(lastID), attentionLayer, linear, residualAddition, norm);
		out.setId(id);
		return out;
	}
}
