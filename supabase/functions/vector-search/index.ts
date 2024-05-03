import "xhr";
import { serve } from "std/http/server.ts";
import { createClient } from "@supabase/supabase-js";
import { codeBlock, oneLine } from "commmon-tags";
import OpenAI from 'https://deno.land/x/openai@v4.24.0/mod.ts'
import { ensureGetEnv } from "../_utils/env.ts";
import { ApplicationError, UserError } from "../_utils/errors.ts";

const SUPABASE_URL = ensureGetEnv("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = ensureGetEnv("SUPABASE_SERVICE_ROLE_KEY");
    
const supabaseClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
    db: { schema: "docs" },
});

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

async function fetchChatBasedOnEmbedding(supabaseClient, questionEmbedding, newChatContent) {
  try {
    const { data, error } = await supabaseClient.rpc('get_last_five_chats_by_embedding', {
      question_embedding: questionEmbedding
    });

    if (error) throw new ApplicationError('RPC Error', error);
    return data || { role: "user", content: newChatContent, chat_embedding: questionEmbedding };
  } catch (error) {
    console.error('Error in fetchChatBasedOnEmbedding:', error);
    throw new ApplicationError('Error fetching or adding chat', error);
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  
  try {
    const searchParams = new URL(req.url).searchParams;
    const query = searchParams.get("query");
    const command = searchParams.get("name") || "General";

    if (!query) throw new UserError("Missing query in request data");

    const OPENAI_KEY = ensureGetEnv("OPENAI_KEY");
    const openai = new OpenAI({
      apiKey: OPENAI_KEY,
    });

    const sanitizedQuery = query.trim().replaceAll("\n", " ");
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: sanitizedQuery,
      encoding_format: "float"
    });
    
    if (embeddingResponse.errors) throw new ApplicationError("Embedding API Error", embeddingResponse.errors);

    const embedding = embeddingResponse.data[0].embedding;
    const pageSections = await supabaseClient.rpc("match_page_sections", {
      embedding,
      match_threshold: 0.18,
      match_count: 1,
      min_content_length: 9,
    });

    let contextText = pageSections.data.map(ps => `${ps.content.trim()}\n---\n`).join('');
    if (pageSections.data.length === 0) {
      const allText = await supabaseClient.rpc("combine_page_contents");
      contextText = allText.data;
    }

    const formatDateUppercase = (date) => {
      return date.toLocaleString('en-US', { month: 'long', day: 'numeric' }).toUpperCase();
    };
    
    const systemPrompt = oneLine`
      You are a direct yet helpful teammate in a group project for an IS Senior Project course. 
      Today's date is ${formatDateUppercase(new Date())}, use this to provide time relevant information in your response. 
      You will be provided with a question and documentation that can be used to reply to the question. 
      If the question is explicitly about technical documentation reply to the best of your ability to assist using the documentation as much as possible, 
      else assume the question is about the syllabus of the IS Senior Project course. 
      You can reference outside information, however, stick to the documentation as much as possible
    `;

    const prompt = codeBlock`
      Documentation:
      ${contextText}
      Question about ${command}:
      ${sanitizedQuery}
      If documentation is empty or not relevant, reply to the best of your ability with outside knowledge.
      Stay concise with your answer, replying specifically to the input prompt.
    `;

    const data = await fetchChatBasedOnEmbedding(supabaseClient, embedding, sanitizedQuery);
    const messages = [{ role: 'system', content: systemPrompt }];
   
    data.forEach(chat => messages.push({ role: chat.role ?? "user", content: chat.content }));
    messages.push({ role: "user", content: prompt });
    const chatCompletion = await openai.chat.completions.create({
      messages,
      model: 'gpt-3.5-turbo',
      stream: false,
      temperature: 1
    });

    await supabaseClient.from('chat_history').insert({
      content: sanitizedQuery, response: chatCompletion.choices[0].message.content, role: 'user', chat_embedding: embedding
    });

    return new Response(chatCompletion.choices[0].message.content, {
      headers: {
        ...corsHeaders,
        "Content-Type": "text/plain",
      },
    });
  } catch (err) {
    return handleError(err);
  }
});

function handleError(err) {
  if (err instanceof UserError || err instanceof ApplicationError) {
    console.error(`${err.name}: ${err.message}`);
    return new Response(JSON.stringify({
      error: err.message,
      data: err.data,
    }), {
      status: err instanceof UserError ? 400 : 500,
      headers: corsHeaders,
    });
  } else {
    console.error('Unexpected error:', err);
    return new Response(JSON.stringify({
      error: "There was an error processing your request",
    }), {
      status: 500,
      headers: corsHeaders,
    });
  }
}
