#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#include <string.h>
#include <stdio.h>

#include <uneural.h>

static int uneural_network_validate_storage(fix16_t *data)
{
	if (*(uint32_t*)data != STORAGE_INIT_MAGIC) {
		return -DATA_STORAGE_UNINITIALIZED;
	}

	return 0;
}

int uneural_network_init_storage(fix16_t *net_data,
                                 ssize_t size)
{
	if (net_data == NULL) {
		return -NULL_ARG;
	}

	memset(net_data, 0x00, size);

	*(uint32_t*)net_data = STORAGE_INIT_MAGIC;

	return 0;
}

ssize_t uneural_network_get_data_requirement(struct uneural_network *n)
{
	/* All non-input neurons require bias + (NUM_INPUTS * weights) bytes of word
	 * aligned storage.  In addition, a 32 bit word holds a magic
	 * keyword indicating that the storage has been initialized. This
	 * function walks the neural net and returns the total amount of
	 * space required (in bytes) to hold all necessary training data */

	/* Start with the size required for the keyword */
	ssize_t total_required = sizeof(uint32_t);

	if (n->output == NULL) {
		return -MISSING_OUTPUT_LAYER;
	}

	if (n->input == NULL) {
		return -MISSING_INPUT_LAYER;
	}

	/* We skip the input layer as no bias or weight are required */
	struct uneural_layer *l = n->input->next;

	while (l != NULL) {
		int temp = (((l->prev->num_neurons * sizeof(fix16_t)) +
			     sizeof(fix16_t) + sizeof(uint32_t)) *
			    l->num_neurons);
		printf("adding %d bytes\n", temp);
		total_required += temp;
		l = l->next;
	}

	return total_required;
}

int uneural_network_data_attach(struct uneural_network *n,
                                fix16_t *data,
                                ssize_t data_size)
{

	fix16_t *start_addr = data;


	/* We do a lot of pointer walking in this library. Make sure the
	 * state storage is aligned to avoid faults on systems where
	 * unaligned access isn't supported */
	if ((intptr_t)data % 4) {
		return -DATA_STORAGE_UNALIGNED;
	}

	if (uneural_network_validate_storage(data) != 0) {
		return -DATA_STORAGE_UNINITIALIZED;
	}

	/* The start of the actual network data starts after the keyword */
	data += sizeof(fix16_t);

	/* We skip the input layer as no bias or weight are required, it
	 * exists simply as a programming convenience */
	struct uneural_layer *l = n->input->next;

	while (l != NULL) {
		/* Make sure L has local neuron storage affixed */
		if (l->neurons == NULL) {
			return -MISSING_NEURON;
		}

		/* Walk the neurons individually and assign them weight and
		 * bias storage */
		for(int i = 0; i < l->num_neurons; i++) {
			l->neurons[i].n_type = (uint32_t*)data;
			data++;
			l->neurons[i].bias = data;
			data++;
			l->neurons[i].weights = data;
			data += (l->prev->num_neurons);
		}

		l = l->next;
        
	}

	n->storage_attached = true;

	/* Calculate the size of the required network data buffer */
	if (data_size < (intptr_t)(data - start_addr)) {
		return -DATA_STORAGE_INSUFFICIENT;
	}
	return 0;
}
