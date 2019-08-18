import os
import ijson
import argparse
import random



def get_context(dialog):
    utterances = dialog['messages-so-far']

    # Create the context
    context = ""
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += msg['utterance'] + ' '# " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context = context.strip() + "\t" + msg['utterance'] + ' '#" __eou__ "
            speaker = msg['speaker']
        else:
            context += msg['utterance'] + ' '#" __eou__ "

    context = context.strip() + "\t"
    return context

def create_train_file(train_file, train_file_out, opt):
    train_file_op = open(train_file_out, "w")
    positive_samples_count = 0
    negative_samples_count = 0

    train_data_handle = open(train_file, 'rb')
    json_data = ijson.items(train_data_handle, 'item')
    for index, entry in enumerate(json_data):
        # row = str(index+1) + "\t"
        context = get_context(entry)
        row = context #+ "\t"

        if len(entry['options-for-correct-answers']) == 0:
            correct_answer = {}
            correct_answer['utterance'] = "None"
            target_id = "NONE"
        else:
            correct_answer = entry['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
        answer = correct_answer['utterance'] #+ " __eou__ "
        answer = answer.strip()
        correct_answer_row = '1\t' + row + answer
        positive_samples_count += 1
        train_file_op.write(correct_answer_row.replace("\n", "") + "\n")

        negative_answers = []
        for i, utterance in enumerate(entry['options-for-next']):
            if utterance['candidate-id'] == target_id:
                continue
            answer = utterance['utterance'] #+ " __eou__ "
            answer = answer.strip()
            negative_answers.append(answer)
        print('train', target_id, len(negative_answers))
        if target_id != "NONE":
            answer = "None"# __eou__"
            negative_answers.append(answer)

        negative_samples = random.sample(negative_answers, opt.neg_pos_ratio)
        assert len(negative_samples) == opt.neg_pos_ratio

        for i in range(len(negative_samples)):
            negative_answer_row = '0\t' + row + negative_samples[i]
            negative_samples_count += 1
            train_file_op.write(negative_answer_row.replace("\n", "") + "\n")

    print("Saved training data to {}".format(train_file_out))
    print("Train - Positive samples count - {}".format(positive_samples_count))
    print("Train - Negative samples count - {}".format(negative_samples_count))
    train_file_op.close()


def create_dev_file(dev_file, dev_file_out):
    dev_file_op = open(dev_file_out, "w")
    positive_samples_count = 0
    negative_samples_count = 0

    dev_data_handle = open(dev_file, 'rb')
    json_data = ijson.items(dev_data_handle, 'item')
    for index, entry in enumerate(json_data):
        # row = str(index+1) + "\t"
        context = get_context(entry)
        row = context #+ "\t"

        if len(entry['options-for-correct-answers']) == 0:
            correct_answer = {}
            correct_answer['utterance'] = "None"
            target_id = "NONE"
        else:
            correct_answer = entry['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
        answer = correct_answer['utterance'] #+ " __eou__ "
        answer = answer.strip()
        correct_answer_row = '1\t' + row + answer
        positive_samples_count += 1
        dev_file_op.write(correct_answer_row.replace("\n", "") + "\n")

        negative_answers = []
        for i, utterance in enumerate(entry['options-for-next']):
            if utterance['candidate-id'] == target_id:
                continue
            answer = utterance['utterance'] #+ " __eou__ "
            answer = answer.strip()
            negative_answers.append(answer)
        print('valid', target_id, len(negative_answers))
        if target_id != "NONE":
            answer = "None"# __eou__"
            negative_answers.append(answer)

        negative_samples = negative_answers

        for i in range(len(negative_samples)):
            negative_answer_row = '0\t' + row + negative_samples[i]
            negative_samples_count += 1
            dev_file_op.write(negative_answer_row.replace("\n", "") + "\n")

    print("Saved valid data to {}".format(dev_file_out))
    print("Valid - Positive samples count - {}".format(positive_samples_count))
    print("Valid - Negative samples count - {}".format(negative_samples_count))
    dev_file_op.close()


def create_test_file(test_file, test_file_out):
    test_file_op = open(test_file_out, "w")
    candidates_count = 0

    test_data_handle = open(test_file, 'rb')
    json_data = ijson.items(test_data_handle, 'item')
    for index, entry in enumerate(json_data):
        entry_id = entry["example-id"]
        row = str(entry_id) + "\t"
        context = get_context(entry)
        row += context + "\t"

        candidates = []
        for i, utterance in enumerate(entry['options-for-next']):
            answer_id = utterance['candidate-id']
            candidates.append(answer_id)
            candidates_count += 1

        candidates.append("NONE")

        candidates = "|".join(candidates)
        row += "NA" + "\t" + candidates + "\t"
        test_file_op.write(row.replace("\n", "") + "\n")

    print("Saved test data to {}".format(test_file_out))
    print("Test - candidates count - {}".format(candidates_count))
    test_file_op.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--train_in', type=str, default="/users5/sychen/DSTC8_DATA/Task_1/ubuntu/task-1.ubuntu.train.json",
                           help="Path to input data file")
    parser.add_argument('-o', '--train_out', type=str, default="/users5/sychen/pinkcom/data/train.txt",
                           help="Path to output train file")
    parser.add_argument('-di', '--dev_in', type=str, default="/users5/sychen/DSTC8_DATA/Task_1/ubuntu/task-1.ubuntu.dev.json",
                           help="Path to dev data file")
    parser.add_argument('-do', '--dev_out', type=str, default="/users5/sychen/pinkcom/data/valid.txt",
                           help="Path to output dev file")
    parser.add_argument('-ti', '--test_in', type=str, default="/users5/sychen/DSTC8_DATA/Task_1/ubuntu/task-1.ubuntu.test.json",
                           help="Path to test data file")
    parser.add_argument('-to', '--test_out', type=str, default="/users5/sychen/pinkcom/data/test.txt",
                           help="Path to output test file")
    #
    # parser.add_argument('-af', '--answers_file', type=str, default="./ans.txt",
    #                     help="Path to write answers file")
    # parser.add_argument('-taf', '--test_answers_file', type=str, default="./",
    #                     help="Path to write test answers file")

    # parser.add_argument('-svp', '--save_vocab_path', type=str, default="./vocab.txt",
    #                     help="Path to save vocabulary txt file")

    parser.add_argument('-r', '--neg_pos_ratio', type=int, default=10,
                        help="Minimum frequency of words in the vocabulary")
    parser.add_argument('-rs', '--random_seed', type=int, default=42,
                        help="Seed for sampling negative training examples")
    opt = parser.parse_args()

    train_file = os.path.join(opt.train_in)
    dev_file = os.path.join(opt.dev_in)
    #test_file = os.path.join(opt.test_in)

    #answers_file = os.path.join(opt.answers_file)
    #test_answers_file = os.path.join(opt.test_answers_file)

    train_file_out = os.path.join(opt.train_out)
    dev_file_out = os.path.join(opt.dev_out)
    #test_file_out = os.path.join(opt.test_out)

    random.seed(opt.random_seed)

    # print("Creating vocabulary...")
    # dialogs = get_dialogs(train_file)
    # input_iter = iter(dialogs)
    # input_iter = create_utterance_iter(input_iter)
    # vocab = create_vocab(input_iter, min_frequency=opt.min_word_frequency)
    # print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary txt file
    # vocab_file = os.path.join(opt.save_vocab_path)
    # write_vocabulary(vocab, vocab_file)

    # Create answers txt file
    # answers = create_answers_file(train_file, dev_file, answers_file)

    ## Once we have the test_set, we need to add test_file here too!
    #answers_test = create_test_answers_file(test_file, test_answers_file)

    # Create train txt file
    create_train_file(train_file, train_file_out, opt)
    # Create dev txt file
    create_dev_file(dev_file, dev_file_out)

    ## Once we have the test_set, we need to add test_file here too!
    # Create test txt file
    #create_test_file(test_file, test_file_out)
