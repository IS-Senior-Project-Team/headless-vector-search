import "xhr";
import { serve } from "std/http/server.ts";
import { createClient } from "@supabase/supabase-js";
import { codeBlock, oneLine } from "commmon-tags";
import GPT3Tokenizer from "gpt3-tokenizer";
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
        match_threshold: 0.78,
        match_count: 10,
        min_content_length: 50,
      }
    );

    if (matchError) {
      throw new ApplicationError("Failed to match page sections", matchError);
    }

    const tokenizer = new GPT3Tokenizer({ type: "gpt3" });
    let tokenCount = 0;
    let contextText = "";

    for (const pageSection of pageSections) {
      const content = pageSection.content;
      const encoded = tokenizer.encode(content);
      tokenCount += encoded.text.length;

      if (tokenCount >= 1500) {
        break;
      }

      contextText += `${content.trim()}\n---\n`;
    }

    const prompt = codeBlock`
      ${oneLine`
        You are an amazing and very helpful teammate in a group project for an IS Senior Project course. 
        Given the following sections from your group project's docs (syllabus, project instructions, etc), answer the question using only that information,
        outputted in markdown format.`}

      Context sections:
      ${contextText}

      Question: """
      ${sanitizedQuery}
      """

      Answer as markdown (including related code snippets if available):
    `;

    const chatCompletion = await openai.chat.completions.create({
      messages: [{ role: 'system', content: prompt }, { role: 'user', content: query }],
      model: 'gpt-3.5-turbo-0125',
      stream: false,
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
