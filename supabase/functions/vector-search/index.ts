import "xhr";
import { serve } from "std/http/server.ts";
import { createClient } from "@supabase/supabase-js";
import { codeBlock, oneLine } from "commmon-tags";
import { Configuration, OpenAIApi } from "openai";
import OpenAI from 'https://deno.land/x/openai@v4.24.0/mod.ts'
import { ensureGetEnv } from "../_utils/env.ts";
import { ApplicationError, UserError } from "../_utils/errors.ts";

const OPENAI_KEY = ensureGetEnv("OPENAI_KEY");
  const openai = new OpenAI({
    apiKey: OPENAI_KEY,
  })
const openAiConfiguration = new Configuration({ apiKey: OPENAI_KEY });
const openaiOld = new OpenAIApi(openAiConfiguration);
const SUPABASE_URL = ensureGetEnv("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = ensureGetEnv("SUPABASE_SERVICE_ROLE_KEY");

const supabaseClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  db: { schema: "docs" },
});


export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  try {
    // Handle CORS
    if (req.method === "OPTIONS") {
      return new Response("ok", { headers: corsHeaders });
    }

    const query = new URL(req.url).searchParams.get("query");

    if (!query) {
      throw new UserError("Missing query in request data");
    }

    const sanitizedQuery = query.trim();

    const embeddingResponse = await openaiOld.createEmbedding({
      model: "text-embedding-ada-002",
      input: sanitizedQuery.replaceAll("\n", " "),
    });

    if (embeddingResponse.status !== 200) {
      throw new ApplicationError(
        "Failed to create embedding for question",
        embeddingResponse
      );
    }

    const [{ embedding }] = embeddingResponse.data.data;

    const { error: matchError, data: pageSections } = await supabaseClient.rpc(
      "match_page_sections",
      {
        embedding,
        match_threshold: 0.4,
        match_count: 5,
        min_content_length: 10,
      }
    );

    if (matchError) {
      throw new ApplicationError("Failed to match page sections", matchError);
    }

    let contextText = "";

    for (const pageSection of pageSections) {
      const content = pageSection.content;
      contextText += `${content.trim()}\n---\n`;
    }

    const systemPrompt = `
      ${oneLine`
      You are a direct yet helpful teammate in a group project for an IS Senior Project course. 
      
      You will be provided with a question and context sections that can be used to reply to the question.
      If a response to the question so it is not explicitly written in the documentation, leave a random joke about AI`}

      Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content.
    `;

    const prompt = `

      Context sections:
      ${contextText}

      Question: """
      ${sanitizedQuery}
      """

      Answer the Question given the Context sections as markdown:
    `;
    const chatCompletion = await openai.chat.completions.create({
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: prompt }],
      model: 'gpt-3.5-turbo',
      stream: false,
      temperature: 1
    })

    // Proxy the streamed SSE response from OpenAI
    return new Response(chatCompletion.choices[0].message.content, {
      headers: {
        ...corsHeaders,
        "Content-Type": "text/plain",
      },
    });
  } catch (err: unknown) {
    if (err instanceof UserError) {
      return Response.json(
        {
          error: err.message,
          data: err.data,
        },
        {
          status: 400,
          headers: corsHeaders,
        }
      );
    } else if (err instanceof ApplicationError) {
      // Print out application errors with their additional data
      console.error(`${err.message}: ${JSON.stringify(err.data)}`);
    } else {
      // Print out unexpected errors as is to help with debugging
      console.error(err);
    }

    // TODO: include more response info in debug environments
    return Response.json(
      {
        error: "There was an error processing your request",
      },
      {
        status: 500,
        headers: corsHeaders,
      }
    );
  }
});
