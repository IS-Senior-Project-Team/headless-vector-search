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
    const OPENAI_KEY = ensureGetEnv("OPENAI_KEY");
    const openai = new OpenAI({
      apiKey: OPENAI_KEY,
    })
    
    const sanitizedQuery = query.trim();

    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: sanitizedQuery.replaceAll("\n", " "),
      encoding_format: "float"
    });

    // if (embeddingResponse.status !== 200) {
    //   throw new ApplicationError(
    //     "Failed to create embedding for question",
    //     embeddingResponse.data[0].embedding
    //   );
    // }
    
    const embedding = embeddingResponse.data[0].embedding;
    const { error: matchError, data: pageSections } = await supabaseClient.rpc(
      "match_page_sections",
      {
        embedding,
        match_threshold: 0.18,
        match_count: 1,
        min_content_length: 9,
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
    if(pageSections.length == 0) {
      const { error: matchError, data: allText } = await supabaseClient.rpc("combine_page_contents");
      contextText = String(allText);
    }
    const formatDateUppercase = (date) => {
      const month = date.toLocaleString('en-US', { month: 'long' }).toUpperCase();
      const day = date.getDate();
    
      return `${month} ${day}`;
    };
    
    const systemPrompt = `
      ${oneLine`
      You are a direct yet helpful teammate in a group project for an IS Senior Project course. 
      Today's date is ${formatDateUppercase(new Date())}, use this to provide time relevant information in your response
      You will be provided with a question and documentation that can be used to reply to the question.
      If a answer to the question is not explicitly written in the documentation, leave a random joke about AI`}
    `;

    const prompt = `

      Documentation: """
      ${contextText}
      """
      Question: """
      ${sanitizedQuery}
      """

      Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content:`;
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
