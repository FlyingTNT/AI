package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersModel;

public class Decoder extends Layer {

	Encoder encoder;
	
	AttentionLayer maskedAttention;
	ResidualAddition maskedAttentionResidual;
	NormLayer maskedAttentionNorm;
	AttentionLayer attention;
	ResidualAddition attentionResidual;
	NormLayer attentionNorm;
	StandardLayer linear;
	ResidualAddition linearResidual;
	NormLayer linearNorm;
	
	Layer[] layers;
	
	public Decoder(Layer last, Encoder encoder, int heads, boolean isFirst, boolean masking)
	{
		super(last, last.outputs);
		maskedAttention = new AttentionLayer(last, last, last, heads, masking, isFirst);
		maskedAttentionResidual = new ResidualAddition(maskedAttention, last);
		maskedAttentionNorm = new NormLayer(maskedAttentionResidual);
		attention = new AttentionLayer(encoder, encoder, maskedAttentionNorm, heads, masking, false);
		attentionResidual = new ResidualAddition(attention, maskedAttentionNorm);
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, last.outputs, Activation.RELU);
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		this.encoder = encoder;
		layers = new Layer[]{maskedAttention, maskedAttentionResidual, maskedAttentionNorm,
									 attention, attentionResidual, attentionNorm,
									 linear, linearResidual, linearNorm};
	}
	
	private Decoder(Layer last, Encoder encoder, AttentionLayer maskedAttention, ResidualAddition maskedResidualAddition, NormLayer maskedNorm, AttentionLayer attention, ResidualAddition attentionResidual, NormLayer attentionNorm, StandardLayer linear)
	{
		super(last, last.outputs);
		this.maskedAttention = maskedAttention;
		this.maskedAttentionResidual = maskedResidualAddition;
		this.maskedAttentionNorm = maskedNorm;
		this.attention = attention;
		this.attentionResidual = attentionResidual;
		this.attentionNorm = attentionNorm;
		this.linear = linear;
		this.linearResidual = new ResidualAddition(linear, attentionNorm);
		this.linearNorm = new NormLayer(linearResidual);
		this.encoder = encoder;
		this.layers = new Layer[]{maskedAttention, maskedAttentionResidual, maskedAttentionNorm,
									 attention, attentionResidual, attentionNorm,
									 linear, linearResidual, linearNorm};
	}
	
	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		for(Layer layer : layers)
			layer.activation(null);
		return linearNorm.lastActivation;
	}

	@Override
	public void backprop() {
		linearNorm.backprop();
		linearResidual.backprop();
		linear.backprop();
		attentionNorm.backprop();
		attentionResidual.backprop();
		attention.backprop();
		maskedAttentionNorm.backprop();
		maskedAttentionResidual.backprop();
		maskedAttention.backprop();
	}

	@Override
	public String name() {
		return "Decoder";
	}

	@Override
	public void reportGradient(SimpleMatrix gradient) {
		linearNorm.reportGradient(gradient);
	}
	
	@Override
	public SimpleMatrix getLastActivation() {
		return linearNorm.getLastActivation();
	}
	
	@Override
	public void setModel(LayersModel model) {
		super.setModel(model);
		for(Layer layer : layers)
			layer.setModel(model);
	}
	
	@Override
	public boolean[] getMasks() {
		return linearNorm.getMasks();
	}
	
	public void setMasking(boolean masking)
	{
		maskedAttention.setMasking(masking);
		attention.setMasking(masking);
	}
	
	public void setInference(boolean inf)
	{
		maskedAttention.decoder = !inf;
	}
	
	@Override
	public String stringify() {
		StringBuilder builder = new StringBuilder();
		builder.append(getId() + " " + lastLayer.getId() + " " + encoder.getId() + " " + maskedAttentionNorm.getId() + " " + attentionNorm.getId() + "\n");
		builder.append(maskedAttention.stringify());
		builder.append("\n##\n");
		builder.append(attention.stringify());
		builder.append("\n##\n");
		builder.append(linear.stringify());
		builder.append("\n##\n");
		return builder.toString();
	}

	public static Decoder load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		int encoderID = scanner.nextInt();
		int maskedNormID = scanner.nextInt();
		int normID = scanner.nextInt();
		scanner.useDelimiter("##");
		AttentionLayer maskedAttentionLayer = AttentionLayer.load(scanner.next(), model, position);
		model.reportLayer(maskedAttentionLayer);
		
		ResidualAddition maskedResidualAddition = new ResidualAddition(maskedAttentionLayer, model.getLayerByID(lastID));
		NormLayer maskedNorm = new NormLayer(maskedResidualAddition);
		maskedNorm.setId(maskedNormID);
		model.reportLayer(maskedNorm);
		
		AttentionLayer attentionLayer = AttentionLayer.load(scanner.next(), model, position);
		model.reportLayer(attentionLayer);
		
		ResidualAddition residualAddition = new ResidualAddition(attentionLayer, maskedNorm);
		NormLayer norm = new NormLayer(residualAddition);
		norm.setId(normID);
		model.reportLayer(norm);	
		
		StandardLayer linear = StandardLayer.load(scanner.next(), model, position);
		scanner.close();
		Decoder out = new Decoder(model.getLayerByID(lastID), (Encoder)model.getLayerByID(encoderID), maskedAttentionLayer, maskedResidualAddition, maskedNorm, attentionLayer, residualAddition, norm, linear);
		out.setId(id);
		return out;
	}
}
