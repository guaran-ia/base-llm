This files describes issues found in the RTT experiments.

### [OSS-GPT-20B](https://huggingface.co/blog/welcome-openai-gpt-oss)

The model `oss-gpt` has been tried and abandonen after several attempts to 
produce translations. The main **limitations** are the following.

* The model is explicitly tuned to “think first” before producing final output.
* For some sentences (even short ones), it expends all token budget reasoning and never generates the translation.
* Deterministic generation (`do_sample=False`) only makes the reasoning deterministic; it doesn't guarantee the final output appears.
* Even with `temperature=0.0` and low `max_new_tokens`, the model may or may not reach final output, depending on its internal token scoring.
This explains why some inputs work and others stall.

#### Unsuccessfull attempts

* Reduce `max_new_tokens` risked to limit reasoning produce truncated outputs.
* Add `stop tokens` can make the model to stop reasoning earlier, losing the translation.
* Prepended final channel token (`>|final|<`) can produce infinite loop because 
the model always expects to start the final channel itself.
* Low `reasoning_effort` helped a little but doesn't guarantee skipping reasoning.
* Explicit instructions ("no reasoning") are ignored if the sentence triggers uncertainty or complexity internally.

#### Conclusions

The model cannot guarantee deterministic translation output. All the tricks tried 
(e.g., stop tokens, prepending final channel, reducing reasoning, max tokens) 
always had edge cases where it failed.
