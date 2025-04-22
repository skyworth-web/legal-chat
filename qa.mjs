import * as dotenv from 'dotenv';
dotenv.config();

import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { ChatOpenAI } from '@langchain/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';
import { match } from 'assert';

let qaChain;

const generateProvId = (part) => {
  console.log(part);
  const matches = part.match(/^(\d+)([A-Za-z]*)/);
  if (!matches) throw new Error('Invalid part format');
  
  const base = '1' + matches[1];
  const suffix = matches[2].toUpperCase();
  
  return `P${base}${suffix}`;
};

export async function initLangchainQA() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  const VECTOR_DIR = path.join(__dirname, 'vectorstore');

  // Load if vector store already exists
  let vectorStore;
  if (fs.existsSync(path.join(VECTOR_DIR, 'docstore.json'))) {
    console.log('🔁 Loading cached vector store...');
    vectorStore = await HNSWLib.load(VECTOR_DIR, new OpenAIEmbeddings());
  } else {
    console.log('🧠 No vector store found. Creating new one...');

    const sources = [
      { file: 'CoA1967.pdf', label: 'Companies Act 1967' },
      { file: 'SFA2001.pdf', label: 'Securities and Futures Act 2001' }
    ];

    const allDocs = [];
    for (const { file, label } of sources) {
      const loader = new PDFLoader(path.join(__dirname, 'documents', file));
      const docs = await loader.load();
      docs.forEach(doc => {
        doc.metadata.source = label;
      });
      allDocs.push(...docs);
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 100,
    });

    const splitDocs = await splitter.splitDocuments(allDocs);

    vectorStore = await HNSWLib.fromDocuments(splitDocs, new OpenAIEmbeddings());
    await vectorStore.save(VECTOR_DIR);
    console.log('✅ Vector store saved to disk.');
  }

  const customPrompt = PromptTemplate.fromTemplate(`
You are a helpful legal assistant specializing in Singapore law, particularly the Companies Act 1967 and the Securities and Futures Act 2001.

Respond clearly and practically to the user’s legal question. If the question is general (e.g. "How do I open a company or restaurant in Singapore?"),
infer the appropriate legal steps under the Companies Act or SFA, and explain them.

Always cite the law when applicable. If unsure, say you don't know rather than guessing.

Your answer should end with:
"Section X (Part Y) of [Act]" Follow this format /Section ([\dA-Za-z]+) \(Part ([\dA-Za-z]+)\) of (CoA1967|SFA2001)/ (e.g., Section 32 (Part 2) of SFA2001, Section 46A (Part 2A) of SFA2001, Section 203A (Part 6) of CoA1967)"

---------------------
{context}
---------------------
Question: {question}
Answer:`);

  qaChain = RetrievalQAChain.fromLLM(
    new ChatOpenAI({ modelName: 'gpt-4-turbo', temperature: 0 }),
    vectorStore.asRetriever(),
    {
      returnSourceDocuments: true,
      prompt: customPrompt,
    }
  );
}

export async function askLegalQuestion(question) {
  if (!qaChain) await initLangchainQA();
  const response = await qaChain.call({ query: question });
  //const sources = response.sourceDocuments?.map(doc => `🔹 ${doc.metadata.source}`).join('\n') || 'No source found';
  const regex = /Section ([\dA-Za-z]+) \(Part ([\dA-Za-z]+)\) of (CoA1967|SFA2001)/gi;
  const matches = regex.exec(response.text);
  let section, part, act, sources, provId;
  if (matches) {
    section = matches[1];  // "12A" (Group 1)
    part = matches[2];     // "3B" (Group 2)
    act = matches[3];      // "SFA2001" (Group 3)
    provId = generateProvId(part);
    sources = `[Section ${section}](https://sso.agc.gov.sg/Act/${act}?ProvIds=${provId}-#pr${section}-)`;
  }
  return `${response.text.trim()}\n\n📚 Sources:\n${sources}`;
}